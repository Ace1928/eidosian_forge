from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.storagegrid.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp import SGRestAPI
class NetAppSgGatherInfo(object):
    """ Class with gather info methods """

    def __init__(self):
        """
        Parse arguments, setup variables, check parameters and ensure
        request module is installed.
        """
        self.argument_spec = netapp_utils.na_storagegrid_host_argument_spec()
        self.argument_spec.update(dict(gather_subset=dict(default=['all'], type='list', elements='str', required=False), parameters=dict(type='dict', required=False)))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = SGRestAPI(self.module)

    def get_subset_info(self, gather_subset_info):
        """
        Gather StorageGRID information for the given subset using REST APIs
        Input for REST APIs call : (api, data)
        return gathered_sg_info
        """
        api = gather_subset_info['api_call']
        data = {}
        if self.parameters.get('parameters'):
            for each in self.parameters['parameters']:
                data[each] = self.parameters['parameters'][each]
        gathered_sg_info, error = self.rest_api.get(api, data)
        if error:
            self.module.fail_json(msg=error)
        else:
            return gathered_sg_info
        return None

    def convert_subsets(self):
        """ Convert an info to the REST API """
        info_to_rest_mapping = {'grid_accounts_info': 'grid/accounts', 'grid_alarms_info': 'grid/alarms', 'grid_audit_info': 'grid/audit', 'grid_compliance_global_info': 'grid/compliance-global', 'grid_config_info': 'grid/config', 'grid_config_management_info': 'grid/config/management', 'grid_config_product_version_info': 'grid/config/product-version', 'grid_deactivated_features_info': 'grid/deactivated-features', 'grid_dns_servers_info': 'grid/dns-servers', 'grid_domain_names_info': 'grid/domain-names', 'grid_ec_profiles_info': 'grid/ec-profiles', 'grid_expansion_info': 'grid/expansion', 'grid_expansion_nodes_info': 'grid/expansion/nodes', 'grid_expansion_sites_info': 'grid/expansion/sites', 'grid_grid_networks_info': 'grid/grid-networks', 'grid_groups_info': 'grid/groups', 'grid_health_info': 'grid/health', 'grid_health_topology_info': 'grid/health/topology', 'grid_identity_source_info': 'grid/identity-source', 'grid_ilm_criteria_info': 'grid/ilm-criteria', 'grid_ilm_policies_info': 'grid/ilm-policies', 'grid_ilm_rules_info': 'grid/ilm-rules', 'grid_license_info': 'grid/license', 'grid_management_certificate_info': 'grid/management-certificate', 'grid_ntp_servers_info': 'grid/ntp-servers', 'grid_recovery_available_nodes_info': 'grid/recovery/available-nodes', 'grid_recovery_info': 'grid/recovery', 'grid_regions_info': 'grid/regions', 'grid_schemes_info': 'grid/schemes', 'grid_snmp_info': 'grid/snmp', 'grid_storage_api_certificate_info': 'grid/storage-api-certificate', 'grid_untrusted_client_network_info': 'grid/untrusted-client-network', 'grid_users_info': 'grid/users', 'grid_users_root_info': 'grid/users/root', 'versions_info': 'versions'}
        subsets = []
        for subset in self.parameters['gather_subset']:
            if subset in info_to_rest_mapping:
                if info_to_rest_mapping[subset] not in subsets:
                    subsets.append(info_to_rest_mapping[subset])
            elif subset not in subsets:
                subsets.append(subset)
        return subsets

    def apply(self):
        """ Perform pre-checks, call functions and exit """
        result_message = dict()
        get_sg_subset_info = {'grid/accounts': {'api_call': 'api/v3/grid/accounts'}, 'grid/alarms': {'api_call': 'api/v3/grid/alarms'}, 'grid/audit': {'api_call': 'api/v3/grid/audit'}, 'grid/compliance-global': {'api_call': 'api/v3/grid/compliance-global'}, 'grid/config': {'api_call': 'api/v3/grid/config'}, 'grid/config/management': {'api_call': 'api/v3/grid/config/management'}, 'grid/config/product-version': {'api_call': 'api/v3/grid/config/product-version'}, 'grid/deactivated-features': {'api_call': 'api/v3/grid/deactivated-features'}, 'grid/dns-servers': {'api_call': 'api/v3/grid/dns-servers'}, 'grid/domain-names': {'api_call': 'api/v3/grid/domain-names'}, 'grid/ec-profiles': {'api_call': 'api/v3/grid/ec-profiles'}, 'grid/expansion': {'api_call': 'api/v3/grid/expansion'}, 'grid/expansion/nodes': {'api_call': 'api/v3/grid/expansion/nodes'}, 'grid/expansion/sites': {'api_call': 'api/v3/grid/expansion/sites'}, 'grid/grid-networks': {'api_call': 'api/v3/grid/grid-networks'}, 'grid/groups': {'api_call': 'api/v3/grid/groups'}, 'grid/health': {'api_call': 'api/v3/grid/health'}, 'grid/health/topology': {'api_call': 'api/v3/grid/health/topology'}, 'grid/identity-source': {'api_call': 'api/v3/grid/identity-source'}, 'grid/ilm-criteria': {'api_call': 'api/v3/grid/ilm-criteria'}, 'grid/ilm-policies': {'api_call': 'api/v3/grid/ilm-policies'}, 'grid/ilm-rules': {'api_call': 'api/v3/grid/ilm-rules'}, 'grid/license': {'api_call': 'api/v3/grid/license'}, 'grid/management-certificate': {'api_call': 'api/v3/grid/management-certificate'}, 'grid/ntp-servers': {'api_call': 'api/v3/grid/ntp-servers'}, 'grid/recovery/available-nodes': {'api_call': 'api/v3/grid/recovery/available-nodes'}, 'grid/recovery': {'api_call': 'api/v3/grid/recovery'}, 'grid/regions': {'api_call': 'api/v3/grid/regions'}, 'grid/schemes': {'api_call': 'api/v3/grid/schemes'}, 'grid/snmp': {'api_call': 'api/v3/grid/snmp'}, 'grid/storage-api-certificate': {'api_call': 'api/v3/grid/storage-api-certificate'}, 'grid/untrusted-client-network': {'api_call': 'api/v3/grid/untrusted-client-network'}, 'grid/users': {'api_call': 'api/v3/grid/users'}, 'grid/users/root': {'api_call': 'api/v3/grid/users/root'}, 'versions': {'api_call': 'api/v3/versions'}}
        if 'all' in self.parameters['gather_subset']:
            self.parameters['gather_subset'] = sorted(get_sg_subset_info.keys())
        converted_subsets = self.convert_subsets()
        for subset in converted_subsets:
            try:
                specified_subset = get_sg_subset_info[subset]
            except KeyError:
                self.module.fail_json(msg='Specified subset %s not found, supported subsets are %s' % (subset, list(get_sg_subset_info.keys())))
            result_message[subset] = self.get_subset_info(specified_subset)
        self.module.exit_json(changed='False', sg_info=result_message)