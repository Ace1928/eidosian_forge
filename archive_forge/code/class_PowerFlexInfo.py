from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import re
class PowerFlexInfo(object):
    """Class with Info operations"""
    filter_mapping = {'equal': 'eq', 'contains': 'co'}

    def __init__(self):
        """ Define all parameters required by this module"""
        self.module_params = utils.get_powerflex_gateway_host_parameters()
        self.module_params.update(get_powerflex_info_parameters())
        self.filter_keys = sorted([k for k in self.module_params['filters']['options'].keys() if 'filter' in k])
        ' initialize the ansible module '
        self.module = AnsibleModule(argument_spec=self.module_params, supports_check_mode=True)
        utils.ensure_required_libs(self.module)
        try:
            self.powerflex_conn = utils.get_powerflex_gateway_host_connection(self.module.params)
            LOG.info('Got the PowerFlex system connection object instance')
            LOG.info('The check_mode flag %s', self.module.check_mode)
        except Exception as e:
            LOG.error(str(e))
            self.module.fail_json(msg=str(e))

    def get_api_details(self):
        """ Get api details of the array """
        try:
            LOG.info('Getting API details ')
            api_version = self.powerflex_conn.system.api_version()
            return api_version
        except Exception as e:
            msg = 'Get API details from Powerflex array failed with error %s' % str(e)
            LOG.error(msg)
            self.module.fail_json(msg=msg)

    def get_array_details(self):
        """ Get system details of a powerflex array """
        try:
            LOG.info('Getting array details ')
            entity_list = ['addressSpaceUsage', 'authenticationMethod', 'capacityAlertCriticalThresholdPercent', 'capacityAlertHighThresholdPercent', 'capacityTimeLeftInDays', 'cliPasswordAllowed', 'daysInstalled', 'defragmentationEnabled', 'enterpriseFeaturesEnabled', 'id', 'installId', 'isInitialLicense', 'lastUpgradeTime', 'managementClientSecureCommunicationEnabled', 'maxCapacityInGb', 'mdmCluster', 'mdmExternalPort', 'mdmManagementPort', 'mdmSecurityPolicy', 'showGuid', 'swid', 'systemVersionName', 'tlsVersion', 'upgradeState']
            sys_list = self.powerflex_conn.system.get()
            sys_details_list = []
            for sys in sys_list:
                sys_details = {}
                for entity in entity_list:
                    if entity in sys.keys():
                        sys_details.update({entity: sys[entity]})
                if sys_details:
                    sys_details_list.append(sys_details)
            return sys_details_list
        except Exception as e:
            msg = 'Get array details from Powerflex array failed with error %s' % str(e)
            LOG.error(msg)
            self.module.fail_json(msg=msg)

    def get_sdc_list(self, filter_dict=None):
        """ Get the list of sdcs on a given PowerFlex storage system """
        try:
            LOG.info('Getting SDC list ')
            if filter_dict:
                sdc = self.powerflex_conn.sdc.get(filter_fields=filter_dict)
            else:
                sdc = self.powerflex_conn.sdc.get()
            return result_list(sdc)
        except Exception as e:
            msg = 'Get SDC list from powerflex array failed with error %s' % str(e)
            LOG.error(msg)
            self.module.fail_json(msg=msg)

    def get_sds_list(self, filter_dict=None):
        """ Get the list of sdses on a given PowerFlex storage system """
        try:
            LOG.info('Getting SDS list ')
            if filter_dict:
                sds = self.powerflex_conn.sds.get(filter_fields=filter_dict)
            else:
                sds = self.powerflex_conn.sds.get()
            return result_list(sds)
        except Exception as e:
            msg = 'Get SDS list from powerflex array failed with error %s' % str(e)
            LOG.error(msg)
            self.module.fail_json(msg=msg)

    def get_pd_list(self, filter_dict=None):
        """ Get the list of Protection Domains on a given PowerFlex
            storage system """
        try:
            LOG.info('Getting protection domain list ')
            if filter_dict:
                pd = self.powerflex_conn.protection_domain.get(filter_fields=filter_dict)
            else:
                pd = self.powerflex_conn.protection_domain.get()
            return result_list(pd)
        except Exception as e:
            msg = 'Get protection domain list from powerflex array failed with error %s' % str(e)
            LOG.error(msg)
            self.module.fail_json(msg=msg)

    def get_storage_pool_list(self, filter_dict=None):
        """ Get the list of storage pools on a given PowerFlex storage
            system """
        try:
            LOG.info('Getting storage pool list ')
            if filter_dict:
                pool = self.powerflex_conn.storage_pool.get(filter_fields=filter_dict)
            else:
                pool = self.powerflex_conn.storage_pool.get()
            if pool:
                statistics_map = self.powerflex_conn.utility.get_statistics_for_all_storagepools()
                list_of_pool_ids_in_statistics = statistics_map.keys()
                for item in pool:
                    item['statistics'] = statistics_map[item['id']] if item['id'] in list_of_pool_ids_in_statistics else {}
            return result_list(pool)
        except Exception as e:
            msg = 'Get storage pool list from powerflex array failed with error %s' % str(e)
            LOG.error(msg)
            self.module.fail_json(msg=msg)

    def get_replication_consistency_group_list(self, filter_dict=None):
        """ Get the list of replication consistency group on a given PowerFlex storage
            system """
        try:
            LOG.info('Getting replication consistency group list ')
            if filter_dict:
                rcgs = self.powerflex_conn.replication_consistency_group.get(filter_fields=filter_dict)
            else:
                rcgs = self.powerflex_conn.replication_consistency_group.get()
            if rcgs:
                api_version = self.powerflex_conn.system.get()[0]['mdmCluster']['master']['versionInfo']
                statistics_map = self.powerflex_conn.replication_consistency_group.get_all_statistics(utils.is_version_less_than_3_6(api_version))
                list_of_rcg_ids_in_statistics = statistics_map.keys()
                for rcg in rcgs:
                    rcg.pop('links', None)
                    rcg['statistics'] = statistics_map[rcg['id']] if rcg['id'] in list_of_rcg_ids_in_statistics else {}
                return result_list(rcgs)
        except Exception as e:
            msg = 'Get replication consistency group list from powerflex array failed with error %s' % str(e)
            LOG.error(msg)
            self.module.fail_json(msg=msg)

    def get_replication_pair_list(self, filter_dict=None):
        """ Get the list of replication pairs on a given PowerFlex storage
            system """
        try:
            LOG.info('Getting replication pair list ')
            if filter_dict:
                pairs = self.powerflex_conn.replication_pair.get(filter_fields=filter_dict)
            else:
                pairs = self.powerflex_conn.replication_pair.get()
            if pairs:
                for pair in pairs:
                    pair.pop('links', None)
                    local_volume = self.powerflex_conn.volume.get(filter_fields={'id': pair['localVolumeId']})
                    if local_volume:
                        pair['localVolumeName'] = local_volume[0]['name']
                    pair['replicationConsistencyGroupName'] = self.powerflex_conn.replication_consistency_group.get(filter_fields={'id': pair['replicationConsistencyGroupId']})[0]['name']
                    pair['statistics'] = self.powerflex_conn.replication_pair.get_statistics(pair['id'])
                return pairs
        except Exception as e:
            msg = 'Get replication pair list from powerflex array failed with error %s' % str(e)
            LOG.error(msg)
            self.module.fail_json(msg=msg)

    def get_volumes_list(self, filter_dict=None):
        """ Get the list of volumes on a given PowerFlex storage
            system """
        try:
            LOG.info('Getting volumes list ')
            if filter_dict:
                volumes = self.powerflex_conn.volume.get(filter_fields=filter_dict)
            else:
                volumes = self.powerflex_conn.volume.get()
            if volumes:
                statistics_map = self.powerflex_conn.utility.get_statistics_for_all_volumes()
                list_of_vol_ids_in_statistics = statistics_map.keys()
                for item in volumes:
                    item['statistics'] = statistics_map[item['id']] if item['id'] in list_of_vol_ids_in_statistics else {}
            return result_list(volumes)
        except Exception as e:
            msg = 'Get volumes list from powerflex array failed with error %s' % str(e)
            LOG.error(msg)
            self.module.fail_json(msg=msg)

    def get_snapshot_policy_list(self, filter_dict=None):
        """ Get the list of snapshot schedules on a given PowerFlex storage
            system """
        try:
            LOG.info('Getting snapshot policies list ')
            if filter_dict:
                snapshot_policies = self.powerflex_conn.snapshot_policy.get(filter_fields=filter_dict)
            else:
                snapshot_policies = self.powerflex_conn.snapshot_policy.get()
            if snapshot_policies:
                statistics_map = self.powerflex_conn.utility.get_statistics_for_all_snapshot_policies()
                list_of_snap_pol_ids_in_statistics = statistics_map.keys()
                for item in snapshot_policies:
                    item['statistics'] = statistics_map[item['id']] if item['id'] in list_of_snap_pol_ids_in_statistics else {}
            return result_list(snapshot_policies)
        except Exception as e:
            msg = 'Get snapshot policies list from powerflex array failed with error %s' % str(e)
            LOG.error(msg)
            self.module.fail_json(msg=msg)

    def get_devices_list(self, filter_dict=None):
        """ Get the list of devices on a given PowerFlex storage
            system """
        try:
            LOG.info('Getting device list ')
            if filter_dict:
                devices = self.powerflex_conn.device.get(filter_fields=filter_dict)
            else:
                devices = self.powerflex_conn.device.get()
            return result_list(devices)
        except Exception as e:
            msg = 'Get device list from powerflex array failed with error %s' % str(e)
            LOG.error(msg)
            self.module.fail_json(msg=msg)

    def get_fault_sets_list(self, filter_dict=None):
        """ Get the list of fault sets on a given PowerFlex storage
            system """
        try:
            LOG.info('Getting fault set list ')
            filter_pd = []
            if filter_dict:
                if 'protectionDomainName' in filter_dict.keys():
                    filter_pd = filter_dict['protectionDomainName']
                    del filter_dict['protectionDomainName']
                fault_sets = self.powerflex_conn.fault_set.get(filter_fields=filter_dict)
            else:
                fault_sets = self.powerflex_conn.fault_set.get()
            fault_set_final = []
            if fault_sets:
                for fault_set in fault_sets:
                    fault_set['protectionDomainName'] = Configuration(self.powerflex_conn, self.module).get_protection_domain(protection_domain_id=fault_set['protectionDomainId'])['name']
                    fault_set['SDS'] = Configuration(self.powerflex_conn, self.module).get_associated_sds(fault_set_id=fault_set['id'])
                    fault_set_final.append(fault_set)
            fault_sets = []
            for fault_set in fault_set_final:
                if fault_set['protectionDomainName'] in filter_pd:
                    fault_sets.append(fault_set)
            if len(filter_pd) != 0:
                return result_list(fault_sets)
            return result_list(fault_set_final)
        except Exception as e:
            msg = 'Get fault set list from powerflex array failed with error %s' % str(e)
            LOG.error(msg)
            self.module.fail_json(msg=msg)

    def get_managed_devices_list(self):
        """ Get the list of managed devices on a given PowerFlex Manager system """
        try:
            LOG.info('Getting managed devices list ')
            devices = self.powerflex_conn.managed_device.get(filters=self.populate_filter_list(), limit=self.get_param_value('limit'), offset=self.get_param_value('offset'), sort=self.get_param_value('sort'))
            return devices
        except Exception as e:
            msg = f'Get managed devices from PowerFlex Manager failed with error {str(e)}'
            return self.handle_error_exit(msg)

    def get_deployments_list(self):
        """ Get the list of deployments on a given PowerFlex Manager system """
        try:
            LOG.info('Getting deployments list ')
            deployments = self.powerflex_conn.deployment.get(filters=self.populate_filter_list(), sort=self.get_param_value('sort'), limit=self.get_param_value('limit'), offset=self.get_param_value('offset'), include_devices=self.get_param_value('include_devices'), include_template=self.get_param_value('include_template'), full=self.get_param_value('full'))
            return deployments
        except Exception as e:
            msg = f'Get deployments from PowerFlex Manager failed with error {str(e)}'
            return self.handle_error_exit(msg)

    def get_service_templates_list(self):
        """ Get the list of service templates on a given PowerFlex Manager system """
        try:
            LOG.info('Getting service templates list ')
            service_templates = self.powerflex_conn.service_template.get(filters=self.populate_filter_list(), sort=self.get_param_value('sort'), offset=self.get_param_value('offset'), limit=self.get_param_value('limit'), full=self.get_param_value('full'), include_attachments=self.get_param_value('include_attachments'))
            return service_templates
        except Exception as e:
            msg = f'Get service templates from PowerFlex Manager failed with error {str(e)}'
            return self.handle_error_exit(msg)

    def handle_error_exit(self, detailed_message):
        match = re.search("displayMessage=([^']+)", detailed_message)
        error_message = match.group(1) if match else detailed_message
        LOG.error(error_message)
        if re.search(ERROR_CODES, detailed_message):
            return []
        self.module.fail_json(msg=error_message)

    def get_param_value(self, param):
        """
        Get the value of the given parameter.
        Args:
            param (str): The parameter to get the value for.
        Returns:
            The value of the parameter if it is different from the default value,
            The value of the parameter if int and greater than 0
            otherwise None.
        """
        if param in ('sort', 'offset', 'limit') and len(self.module.params.get('gather_subset')) > 1:
            return None
        default_value = self.module_params.get(param).get('default')
        param_value = self.module.params.get(param)
        if default_value != param_value and (param_value >= 0 if isinstance(param_value, int) else True):
            return param_value
        return None

    def validate_filter(self, filter_dict):
        """ Validate given filter_dict """
        is_invalid_filter = self.filter_keys != sorted(list(filter_dict))
        if is_invalid_filter:
            msg = "Filter should have all keys: '{0}'".format(', '.join(self.filter_keys))
            LOG.error(msg)
            self.module.fail_json(msg=msg)
        is_invalid_filter = [filter_dict[i] is None for i in filter_dict]
        if True in is_invalid_filter:
            msg = "Filter keys: '{0}' cannot be None".format(self.filter_keys)
            LOG.error(msg)
            self.module.fail_json(msg=msg)

    def populate_filter_list(self):
        """Populate the filter list"""
        if len(self.module.params.get('gather_subset')) > 1:
            return []
        filters = self.module.params.get('filters') or []
        return [f'{self.filter_mapping.get(filter_dict['filter_operator'])},{filter_dict['filter_key']},{filter_dict['filter_value']}' for filter_dict in filters]

    def get_filters(self, filters):
        """Get the filters to be applied"""
        filter_dict = {}
        for item in filters:
            self.validate_filter(item)
            f_op = item['filter_operator']
            if self.filter_mapping.get(f_op) == self.filter_mapping.get('equal'):
                f_key = item['filter_key']
                f_val = item['filter_value']
                if f_key in filter_dict:
                    if isinstance(filter_dict[f_key], list):
                        filter_dict[f_key].append(f_val)
                    else:
                        filter_dict[f_key] = [filter_dict[f_key], f_val]
                else:
                    filter_dict[f_key] = f_val
        return filter_dict

    def validate_subset(self, api_version, subset):
        if float(api_version) < MIN_SUPPORTED_POWERFLEX_MANAGER_VERSION and subset and set(subset).issubset(POWERFLEX_MANAGER_GATHER_SUBSET):
            self.module.exit_json(msg=UNSUPPORTED_SUBSET_FOR_VERSION, skipped=True)

    def perform_module_operation(self):
        """ Perform different actions on info based on user input
            in the playbook """
        filters = self.module.params['filters']
        filter_dict = {}
        if filters:
            filter_dict = self.get_filters(filters)
            LOG.info('filters: %s', filter_dict)
        api_version = self.get_api_details()
        array_details = self.get_array_details()
        sdc = []
        sds = []
        storage_pool = []
        vol = []
        snapshot_policy = []
        protection_domain = []
        device = []
        rcgs = []
        replication_pair = []
        fault_sets = []
        service_template = []
        managed_device = []
        deployment = []
        subset = self.module.params['gather_subset']
        self.validate_subset(api_version, subset)
        if subset is not None:
            if 'sdc' in subset:
                sdc = self.get_sdc_list(filter_dict=filter_dict)
            if 'sds' in subset:
                sds = self.get_sds_list(filter_dict=filter_dict)
            if 'protection_domain' in subset:
                protection_domain = self.get_pd_list(filter_dict=filter_dict)
            if 'storage_pool' in subset:
                storage_pool = self.get_storage_pool_list(filter_dict=filter_dict)
            if 'vol' in subset:
                vol = self.get_volumes_list(filter_dict=filter_dict)
            if 'snapshot_policy' in subset:
                snapshot_policy = self.get_snapshot_policy_list(filter_dict=filter_dict)
            if 'device' in subset:
                device = self.get_devices_list(filter_dict=filter_dict)
            if 'rcg' in subset:
                rcgs = self.get_replication_consistency_group_list(filter_dict=filter_dict)
            if 'replication_pair' in subset:
                replication_pair = self.get_replication_pair_list(filter_dict=filter_dict)
            if 'fault_set' in subset:
                fault_sets = self.get_fault_sets_list(filter_dict=filter_dict)
            if 'managed_device' in subset:
                managed_device = self.get_managed_devices_list()
            if 'service_template' in subset:
                service_template = self.get_service_templates_list()
            if 'deployment' in subset:
                deployment = self.get_deployments_list()
        self.module.exit_json(Array_Details=array_details, API_Version=api_version, SDCs=sdc, SDSs=sds, Storage_Pools=storage_pool, Volumes=vol, Snapshot_Policies=snapshot_policy, Protection_Domains=protection_domain, Devices=device, Replication_Consistency_Groups=rcgs, Replication_Pairs=replication_pair, Fault_Sets=fault_sets, ManagedDevices=managed_device, ServiceTemplates=service_template, Deployments=deployment)