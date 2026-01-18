from copy import deepcopy
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.community.aws.plugins.module_utils.ec2 import BaseEc2Manager
from ansible_collections.community.aws.plugins.module_utils.ec2 import Boto3Mixin
from ansible_collections.community.aws.plugins.module_utils.ec2 import Ec2WaiterFactory
class TransitGatewayVpcAttachmentManager(TGWAttachmentBoto3Mixin, BaseTGWManager):
    TAG_RESOURCE_TYPE = 'transit-gateway-attachment'

    def __init__(self, module, id=None):
        self._subnet_updates = dict()
        super(TransitGatewayVpcAttachmentManager, self).__init__(module=module, id=id)

    def _get_id_params(self, id=None, id_list=False):
        if not id:
            id = self.resource_id
        if not id:
            self.module.fail_json(msg='Attachment identifier parameter missing')
        if id_list:
            return dict(TransitGatewayAttachmentIds=[id])
        return dict(TransitGatewayAttachmentId=id)

    def _extra_error_output(self):
        output = super(TransitGatewayVpcAttachmentManager, self)._extra_error_output()
        if self.resource_id:
            output['TransitGatewayAttachmentId'] = self.resource_id
        return output

    def _filter_immutable_resource_attributes(self, resource):
        resource = super(TransitGatewayVpcAttachmentManager, self)._filter_immutable_resource_attributes(resource)
        resource.pop('TransitGatewayId', None)
        resource.pop('VpcId', None)
        resource.pop('VpcOwnerId', None)
        resource.pop('State', None)
        resource.pop('SubnetIds', None)
        resource.pop('CreationTime', None)
        resource.pop('Tags', None)
        return resource

    def _set_option(self, name, value):
        if value is None:
            return False
        if value:
            value = 'enable'
        else:
            value = 'disable'
        options = deepcopy(self._preupdate_resource.get('Options', dict()))
        options.update(self._resource_updates.get('Options', dict()))
        options[name] = value
        return self._set_resource_value('Options', options)

    def set_dns_support(self, value):
        return self._set_option('DnsSupport', value)

    def set_ipv6_support(self, value):
        return self._set_option('Ipv6Support', value)

    def set_appliance_mode_support(self, value):
        return self._set_option('ApplianceModeSupport', value)

    def set_transit_gateway(self, tgw_id):
        return self._set_resource_value('TransitGatewayId', tgw_id)

    def set_vpc(self, vpc_id):
        return self._set_resource_value('VpcId', vpc_id)

    def set_subnets(self, subnets=None, purge=True):
        if subnets is None:
            return False
        current_subnets = set(self._preupdate_resource.get('SubnetIds', []))
        desired_subnets = set(subnets)
        if not purge:
            desired_subnets = desired_subnets.union(current_subnets)
        subnet_details = self._describe_subnets(SubnetIds=list(desired_subnets))
        vpc_id = self.subnets_to_vpc(desired_subnets, subnet_details)
        self._set_resource_value('VpcId', vpc_id, immutable=True)
        azs = [s.get('AvailabilityZoneId') for s in subnet_details]
        if len(azs) != len(set(azs)):
            self.module.fail_json(msg='Only one attachment subnet per availability zone may be set.', availability_zones=azs, subnets=subnet_details)
        subnets_to_add = list(desired_subnets.difference(current_subnets))
        subnets_to_remove = list(current_subnets.difference(desired_subnets))
        if not subnets_to_remove and (not subnets_to_add):
            return False
        self._subnet_updates = dict(add=subnets_to_add, remove=subnets_to_remove)
        self._set_resource_value('SubnetIds', list(desired_subnets))
        return True

    def subnets_to_vpc(self, subnets, subnet_details=None):
        if not subnets:
            return None
        if subnet_details is None:
            subnet_details = self._describe_subnets(SubnetIds=list(subnets))
        vpcs = [s.get('VpcId') for s in subnet_details]
        if len(set(vpcs)) > 1:
            self.module.fail_json(msg='Attachment subnets may only be in one VPC, multiple VPCs found', vpcs=list(set(vpcs)), subnets=subnet_details)
        return vpcs[0]

    def _do_deletion_wait(self, id=None, **params):
        all_params = self._get_id_params(id=id, id_list=True)
        all_params.update(**params)
        return self._wait_tgw_attachment_deleted(**all_params)

    def _do_creation_wait(self, id=None, **params):
        all_params = self._get_id_params(id=id, id_list=True)
        all_params.update(**params)
        return self._wait_tgw_attachment_available(**all_params)

    def _do_update_wait(self, id=None, **params):
        all_params = self._get_id_params(id=id, id_list=True)
        all_params.update(**params)
        return self._wait_tgw_attachment_available(**all_params)

    def _do_create_resource(self):
        params = self._merge_resource_changes(filter_immutable=False, creation=True)
        response = self._create_vpc_attachment(**params)
        if response:
            self.resource_id = response.get('TransitGatewayAttachmentId', None)
        return response

    def _do_update_resource(self):
        if self._preupdate_resource.get('State', None) == 'pending':
            self._wait_for_creation()
        elif self._preupdate_resource.get('State', None) == 'deleting':
            self.module.fail_json(msg='Deletion in progress, unable to update', route_tables=[self.original_resource])
        updates = self._filter_immutable_resource_attributes(self._resource_updates)
        subnets_to_add = self._subnet_updates.get('add', [])
        subnets_to_remove = self._subnet_updates.get('remove', [])
        if subnets_to_add:
            updates['AddSubnetIds'] = subnets_to_add
        if subnets_to_remove:
            updates['RemoveSubnetIds'] = subnets_to_remove
        if not updates:
            return False
        if self.module.check_mode:
            return True
        updates.update(self._get_id_params(id_list=False))
        self._modify_vpc_attachment(**updates)
        return True

    def get_resource(self):
        return self.get_attachment()

    def delete(self, id=None):
        if id:
            id_params = self._get_id_params(id=id, id_list=True)
            result = self._get_tgw_vpc_attachment(**id_params)
        else:
            result = self._preupdate_resource
        self.updated_resource = dict()
        if not result:
            return False
        if result.get('State') == 'deleting':
            self._wait_for_deletion()
            return False
        if self.module.check_mode:
            self.changed = True
            return True
        id_params = self._get_id_params(id=id, id_list=False)
        result = self._delete_vpc_attachment(**id_params)
        self.changed |= bool(result)
        self._wait_for_deletion()
        return bool(result)

    def list(self, filters=None, id=None):
        params = dict()
        if id:
            params['TransitGatewayAttachmentIds'] = [id]
        if filters:
            params['Filters'] = ansible_dict_to_boto3_filter_list(filters)
        attachments = self._describe_vpc_attachments(**params)
        if not attachments:
            return list()
        return [self._normalize_tgw_attachment(a) for a in attachments]

    def get_attachment(self, id=None):
        id_params = self._get_id_params(id=id, id_list=True)
        id_param = self._get_id_params(id=id, id_list=False)
        result = self._get_tgw_vpc_attachment(**id_params)
        if not result:
            return None
        if not id:
            self._preupdate_resource = deepcopy(result)
        attachment = self._normalize_tgw_attachment(result)
        return attachment

    def _normalize_resource(self, resource):
        return self._normalize_tgw_attachment(resource)