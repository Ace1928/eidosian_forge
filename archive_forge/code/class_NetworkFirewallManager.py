import time
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.arn import parse_aws_arn
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.base import BaseResourceManager
from ansible_collections.community.aws.plugins.module_utils.base import BaseWaiterFactory
from ansible_collections.community.aws.plugins.module_utils.base import Boto3Mixin
from ansible_collections.community.aws.plugins.module_utils.ec2 import BaseEc2Manager
class NetworkFirewallManager(NFFirewallBoto3Mixin, NFPolicyBoto3Mixin, BaseNetworkFirewallManager):
    name = None
    arn = None
    ec2_manager = None
    _subnet_updates = None
    _policy_list_cache = None
    _slow_start_change = False

    def __init__(self, module, name=None, arn=None):
        super().__init__(module)
        self.name = name
        self.arn = arn
        self.ec2_manager = BaseEc2Manager(module=module)
        self._subnet_updates = dict()
        if self.name or self.arn:
            firewall = deepcopy(self.get_firewall())
            self.original_resource = firewall

    def _extra_error_output(self):
        output = super(NetworkFirewallManager, self)._extra_error_output()
        if self.name:
            output['FirewallName'] = self.name
        if self.arn:
            output['FirewallArn'] = self.arn
        return output

    def _get_preupdate_arn(self):
        return self._get_resource_value('FirewallArn')

    def _get_id_params(self, name=None, arn=None):
        if arn:
            return dict(FirewallArn=arn)
        if self.arn:
            return dict(FirewallArn=self.arn)
        if not name:
            name = self.name
        if not name:
            self.module.fail_json(msg='Firewall identifier parameters missing')
        return dict(FirewallName=name)

    def delete(self, name=None, arn=None):
        id_params = self._get_id_params(name=name, arn=arn)
        result = self._get_firewall(**id_params)
        if not result:
            return False
        self.updated_resource = dict()
        firewall_status = self._get_metadata_value('Status', '').upper()
        if firewall_status == 'DELETING':
            self._wait_for_deletion()
            return False
        if self.module.check_mode:
            self.changed = True
            return True
        if 'DeleteProtection' in self._resource_updates:
            self._update_firewall_delete_protection(DeleteProtection=self._resource_updates['DeleteProtection'], **id_params)
        result = self._delete_firewall(**id_params)
        self._wait_for_deletion()
        self.changed |= bool(result)
        return bool(result)

    def list(self, vpc_ids=None):
        params = dict()
        if vpc_ids:
            params['VpcIds'] = vpc_ids
        firewalls = self._list_firewalls(**params)
        if not firewalls:
            return list()
        return [f.get('FirewallArn', None) for f in firewalls]

    def _normalize_firewall(self, firewall):
        if firewall is None:
            return None
        subnets = [s.get('SubnetId') for s in firewall.get('SubnetMappings', [])]
        firewall = self._normalize_boto3_resource(firewall, add_tags=True)
        firewall['subnets'] = subnets
        return firewall

    def _normalize_sync_state_config(self, policy):
        return self._normalize_boto3_resource(policy)

    def _normalize_sync_state(self, state):
        config = {k: self._normalize_sync_state_config(v) for k, v in state.pop('Config', {}).items()}
        state = self._normalize_boto3_resource(state)
        state['config'] = config or {}
        return state

    def _normalize_firewall_metadata(self, firewall_metadata):
        if firewall_metadata is None:
            return None
        states = {k: self._normalize_sync_state(v) for k, v in firewall_metadata.pop('SyncStates', {}).items()}
        metadata = self._normalize_boto3_resource(firewall_metadata, add_tags=False)
        metadata['sync_states'] = states or {}
        return metadata

    def _normalize_firewall_result(self, result):
        if result is None:
            return None
        firewall = self._normalize_firewall(result.get('Firewall', None))
        firewall_metadata = self._normalize_firewall_metadata(result.get('FirewallMetadata', None))
        result = camel_dict_to_snake_dict(result)
        if firewall:
            result['firewall'] = firewall
        if firewall_metadata:
            result['firewall_metadata'] = firewall_metadata
        return result

    def _normalize_resource(self, resource):
        return self._normalize_firewall_result(resource)

    def get_firewall(self, name=None, arn=None):
        id_params = self._get_id_params(name=name, arn=arn)
        result = self._get_firewall(**id_params)
        if not result:
            return None
        firewall = self._normalize_firewall_result(result)
        return firewall

    @property
    def _subnets(self):
        subnet_mappings = self._get_resource_value('SubnetMappings', [])
        subnets = [s.get('SubnetId') for s in subnet_mappings]
        return subnets

    def _subnets_to_vpc(self, subnets, subnet_details=None):
        if not subnets:
            return None
        if not subnet_details:
            subnet_details = self.ec2_manager._describe_subnets(SubnetIds=list(subnets))
        vpcs = [s.get('VpcId') for s in subnet_details]
        if len(set(vpcs)) > 1:
            self.module.fail_json(msg='Firewall subnets may only be in one VPC, multiple VPCs found', vpcs=list(set(vpcs)), subnets=subnet_details)
        return vpcs[0]

    def _format_subnet_mapping(self, subnets):
        if not subnets:
            return []
        return [dict(SubnetId=s) for s in subnets]

    @property
    def _policy_name_cache(self):
        if self._policy_list_cache:
            return self._policy_list_cache
        results = self._list_policies()
        if not results:
            return dict()
        policy_cache = {p.get('Name', None): p.get('Arn', None) for p in results}
        self._policy_list_cache = policy_cache
        return policy_cache

    def _canonicalize_policy(self, name):
        """Iterates through a mixed list of ARNs and Names converting them to
        ARNs.
        """
        arn = None
        if ':' in name:
            arn = name
        else:
            arn = self._policy_name_cache.get(name, None)
            if not arn:
                self.module.fail_json('Unable to fetch ARN for policy', name=name, policy_name_cache=self._policy_name_cache)
        arn_info = parse_aws_arn(arn)
        if not arn_info:
            self.module.fail_json('Unable to parse ARN for policy', arn=arn, arn_info=arn_info)
        arn_type = arn_info['resource'].split('/')[0]
        if arn_type != 'firewall-policy':
            self.module.fail_json('Policy ARN not of expected resource type', name=name, arn=arn, expected_type='firewall-policy', found_type=arn_type)
        return arn

    def set_policy(self, policy):
        if policy is None:
            return False
        current_policy = self._get_resource_value('FirewallPolicyArn', None)
        if current_policy:
            arn_info = parse_aws_arn(current_policy)
            current_name = arn_info['resource'].split('/')[-1]
            if current_name == policy:
                return False
        policy = self._canonicalize_policy(policy)
        return self._set_resource_value('FirewallPolicyArn', policy)

    def set_subnets(self, subnets, purge=True):
        if subnets is None:
            return False
        current_subnets = set(self._subnets)
        desired_subnets = set(subnets)
        if not purge:
            desired_subnets = desired_subnets.union(current_subnets)
        if current_subnets == desired_subnets:
            return False
        subnet_details = self.ec2_manager._describe_subnets(SubnetIds=list(desired_subnets))
        vpc = self._subnets_to_vpc(desired_subnets, subnet_details)
        self._set_resource_value('VpcId', vpc, description='firewall VPC', immutable=True)
        azs = [s.get('AvailabilityZoneId') for s in subnet_details]
        if len(azs) != len(set(azs)):
            self.module.fail_json(msg='Only one subnet per availability zone may set.', availability_zones=azs, subnets=subnet_details)
        subnets_to_add = list(desired_subnets.difference(current_subnets))
        subnets_to_remove = list(current_subnets.difference(desired_subnets))
        self._subnet_updates = dict(add=subnets_to_add, remove=subnets_to_remove)
        self._set_resource_value('SubnetMappings', self._format_subnet_mapping(desired_subnets))
        return True

    def set_policy_change_protection(self, protection):
        return self._set_resource_value('FirewallPolicyChangeProtection', protection)

    def set_subnet_change_protection(self, protection):
        return self._set_resource_value('SubnetChangeProtection', protection)

    def set_delete_protection(self, protection):
        return self._set_resource_value('DeleteProtection', protection)

    def set_description(self, description):
        return self._set_resource_value('Description', description)

    def _do_create_resource(self):
        metadata, resource = self._merge_changes(filter_metadata=False)
        params = metadata
        params.update(self._get_id_params())
        params.update(resource)
        response = self._create_firewall(**params)
        return bool(response)

    def _generate_updated_resource(self):
        metadata, resource = self._merge_changes(filter_metadata=False)
        resource.update(self._get_id_params())
        updated_resource = dict(Firewall=resource, FirewallMetadata=metadata)
        return updated_resource

    def _flush_create(self):
        return super(NetworkFirewallManager, self)._flush_create()

    def _do_update_resource(self):
        resource_updates = self._resource_updates
        if not resource_updates:
            return False
        if self.module.check_mode:
            return True
        id_params = self._get_id_params()
        if 'Description' in resource_updates:
            self._update_firewall_description(Description=resource_updates['Description'], **id_params)
        if 'DeleteProtection' in resource_updates:
            self._update_firewall_delete_protection(DeleteProtection=resource_updates['DeleteProtection'], **id_params)
        if 'FirewallPolicyChangeProtection' in resource_updates:
            if not self._get_resource_value('FirewallPolicyChangeProtection'):
                self._update_firewall_policy_change_protection(FirewallPolicyChangeProtection=resource_updates['FirewallPolicyChangeProtection'], **id_params)
        if 'SubnetChangeProtection' in resource_updates:
            if not self._get_resource_value('SubnetChangeProtection'):
                self._update_subnet_change_protection(SubnetChangeProtection=resource_updates['SubnetChangeProtection'], **id_params)
        if 'SubnetMappings' in resource_updates:
            self._slow_start_change = True
            subnets_to_add = self._subnet_updates.get('add', None)
            subnets_to_remove = self._subnet_updates.get('remove', None)
            if subnets_to_remove:
                self._disassociate_subnets(SubnetIds=subnets_to_remove, **id_params)
            if subnets_to_add:
                subnets_to_add = self._format_subnet_mapping(subnets_to_add)
                self._associate_subnets(SubnetMappings=subnets_to_add, **id_params)
        if 'FirewallPolicyArn' in resource_updates:
            self._slow_start_change = True
            self._associate_firewall_policy(FirewallPolicyArn=resource_updates['FirewallPolicyArn'], **id_params)
        if 'FirewallPolicyChangeProtection' in resource_updates:
            if self._get_resource_value('FirewallPolicyChangeProtection'):
                self._update_firewall_policy_change_protection(FirewallPolicyChangeProtection=resource_updates['FirewallPolicyChangeProtection'], **id_params)
        if 'SubnetChangeProtection' in resource_updates:
            if self._get_resource_value('SubnetChangeProtection'):
                self._update_subnet_change_protection(SubnetChangeProtection=resource_updates['SubnetChangeProtection'], **id_params)
        return True

    def _flush_update(self):
        changed = False
        changed |= self._flush_tagging()
        changed |= super(NetworkFirewallManager, self)._flush_update()
        self._subnet_updates = dict()
        self._slow_start_change = False
        return changed

    def _get_firewall(self, **params):
        result = self._describe_firewall(**params)
        if not result:
            return None
        firewall = result.get('Firewall', None)
        metadata = result.get('FirewallMetadata', None)
        self._preupdate_resource = deepcopy(firewall)
        self._preupdate_metadata = deepcopy(metadata)
        return dict(Firewall=firewall, FirewallMetadata=metadata)

    def get_resource(self):
        return self.get_firewall()

    def _do_creation_wait(self, **params):
        all_params = self._get_id_params()
        all_params.update(params)
        return self._wait_firewall_active(**all_params)

    def _do_deletion_wait(self, **params):
        all_params = self._get_id_params()
        all_params.update(params)
        return self._wait_firewall_deleted(**all_params)

    def _do_update_wait(self, **params):
        if self._slow_start_change:
            time.sleep(4)
        all_params = self._get_id_params()
        all_params.update(params)
        return self._wait_firewall_updated(**all_params)

    def _set_tag_values(self, desired_tags):
        return self._set_resource_value('Tags', ansible_dict_to_boto3_tag_list(desired_tags))

    def _get_tag_values(self):
        return self._get_resource_value('Tags', [])