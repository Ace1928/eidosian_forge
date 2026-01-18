from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
class NetAppCloudManagerCVOGCP:
    """ object initialize and class methods """

    def __init__(self):
        self.use_rest = False
        self.argument_spec = netapp_utils.cloudmanager_host_argument_spec()
        self.argument_spec.update(dict(backup_volumes_to_cbs=dict(required=False, type='bool', default=False), capacity_tier=dict(required=False, type='str', choices=['cloudStorage']), client_id=dict(required=True, type='str'), data_encryption_type=dict(required=False, choices=['GCP'], type='str'), gcp_encryption_parameters=dict(required=False, type='str', no_log=True), enable_compliance=dict(required=False, type='bool', default=False), firewall_rule=dict(required=False, type='str'), gcp_labels=dict(required=False, type='list', elements='dict', options=dict(label_key=dict(type='str', no_log=False), label_value=dict(type='str'))), gcp_service_account=dict(required=True, type='str'), gcp_volume_size=dict(required=False, type='int'), gcp_volume_size_unit=dict(required=False, choices=['GB', 'TB'], type='str'), gcp_volume_type=dict(required=False, choices=['pd-balanced', 'pd-standard', 'pd-ssd'], type='str'), instance_type=dict(required=False, type='str', default='n1-standard-8'), is_ha=dict(required=False, type='bool', default=False), license_type=dict(required=False, type='str', choices=GCP_LICENSE_TYPES, default='capacity-paygo'), mediator_zone=dict(required=False, type='str'), name=dict(required=True, type='str'), network_project_id=dict(required=False, type='str'), node1_zone=dict(required=False, type='str'), node2_zone=dict(required=False, type='str'), nss_account=dict(required=False, type='str'), ontap_version=dict(required=False, type='str', default='latest'), platform_serial_number=dict(required=False, type='str'), platform_serial_number_node1=dict(required=False, type='str'), platform_serial_number_node2=dict(required=False, type='str'), project_id=dict(required=True, type='str'), state=dict(required=False, choices=['present', 'absent'], default='present'), subnet_id=dict(required=False, type='str'), subnet0_node_and_data_connectivity=dict(required=False, type='str'), subnet1_cluster_connectivity=dict(required=False, type='str'), subnet2_ha_connectivity=dict(required=False, type='str'), subnet3_data_replication=dict(required=False, type='str'), svm_password=dict(required=False, type='str', no_log=True), svm_name=dict(required=False, type='str'), tier_level=dict(required=False, type='str', choices=['standard', 'nearline', 'coldline'], default='standard'), use_latest_version=dict(required=False, type='bool', default=True), capacity_package_name=dict(required=False, type='str', choices=['Professional', 'Essential', 'Freemium'], default='Essential'), provided_license=dict(required=False, type='str'), vpc_id=dict(required=True, type='str'), vpc0_firewall_rule_name=dict(required=False, type='str'), vpc0_node_and_data_connectivity=dict(required=False, type='str'), vpc1_cluster_connectivity=dict(required=False, type='str'), vpc1_firewall_rule_name=dict(required=False, type='str'), vpc2_firewall_rule_name=dict(required=False, type='str'), vpc2_ha_connectivity=dict(required=False, type='str'), vpc3_data_replication=dict(required=False, type='str'), vpc3_firewall_rule_name=dict(required=False, type='str'), workspace_id=dict(required=False, type='str'), writing_speed_state=dict(required=False, type='str'), zone=dict(required=True, type='str'), upgrade_ontap_version=dict(required=False, type='bool', default=False), update_svm_password=dict(required=False, type='bool', default=False), subnet_path=dict(required=False, type='str')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, required_one_of=[['refresh_token', 'sa_client_id']], required_together=[['sa_client_id', 'sa_secret_key']], required_if=[['license_type', 'capacity-paygo', ['capacity_package_name']], ['license_type', 'ha-capacity-paygo', ['capacity_package_name']], ['license_type', 'gcp-cot-premium-byol', ['platform_serial_number']], ['license_type', 'gcp-ha-cot-premium-byol', ['platform_serial_number_node1', 'platform_serial_number_node2']]], supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.changeable_params = ['svm_password', 'svm_name', 'tier_level', 'gcp_labels', 'ontap_version', 'instance_type', 'license_type', 'writing_speed_state']
        self.rest_api = CloudManagerRestAPI(self.module)
        self.rest_api.url += self.rest_api.environment_data['CLOUD_MANAGER_HOST']
        self.rest_api.api_root_path = '/occm/api/gcp/%s' % ('ha' if self.parameters['is_ha'] else 'vsa')
        self.headers = {'X-Agent-Id': self.rest_api.format_client_id(self.parameters['client_id'])}

    @staticmethod
    def has_self_link(param):
        return param.startswith(('https://www.googleapis.com/compute/', 'projects/'))

    def create_cvo_gcp(self):
        if self.parameters.get('workspace_id') is None:
            response, msg = self.na_helper.get_tenant(self.rest_api, self.headers)
            if response is None:
                self.module.fail_json(msg)
            self.parameters['workspace_id'] = response
        if self.parameters.get('nss_account') is None:
            if self.parameters.get('platform_serial_number') is not None:
                if not self.parameters['platform_serial_number'].startswith('Eval-'):
                    if self.parameters['license_type'] == 'gcp-cot-premium-byol' or self.parameters['license_type'] == 'gcp-ha-cot-premium-byol':
                        response, msg = self.na_helper.get_nss(self.rest_api, self.headers)
                        if response is None:
                            self.module.fail_json(msg)
                        self.parameters['nss_account'] = response
        if self.parameters['is_ha'] is True and self.parameters['license_type'] == 'capacity-paygo':
            self.parameters['license_type'] == 'ha-capacity-paygo'
        json = {'name': self.parameters['name'], 'region': self.parameters['zone'], 'tenantId': self.parameters['workspace_id'], 'vpcId': self.parameters['vpc_id'], 'gcpServiceAccount': self.parameters['gcp_service_account'], 'gcpVolumeSize': {'size': self.parameters['gcp_volume_size'], 'unit': self.parameters['gcp_volume_size_unit']}, 'gcpVolumeType': self.parameters['gcp_volume_type'], 'svmPassword': self.parameters['svm_password'], 'backupVolumesToCbs': self.parameters['backup_volumes_to_cbs'], 'enableCompliance': self.parameters['enable_compliance'], 'vsaMetadata': {'ontapVersion': self.parameters['ontap_version'], 'licenseType': self.parameters['license_type'], 'useLatestVersion': self.parameters['use_latest_version'], 'instanceType': self.parameters['instance_type']}}
        if self.parameters['is_ha'] is False:
            if self.parameters.get('writing_speed_state') is None:
                self.parameters['writing_speed_state'] = 'NORMAL'
            json.update({'writingSpeedState': self.parameters['writing_speed_state'].upper()})
        if self.parameters.get('data_encryption_type') is not None and self.parameters['data_encryption_type'] == 'GCP':
            json.update({'dataEncryptionType': self.parameters['data_encryption_type']})
            if self.parameters.get('gcp_encryption_parameters') is not None:
                json.update({'gcpEncryptionParameters': {'key': self.parameters['gcp_encryption_parameters']}})
        if self.parameters.get('provided_license') is not None:
            json['vsaMetadata'].update({'providedLicense': self.parameters['provided_license']})
        if not self.parameters['license_type'].endswith('capacity-paygo'):
            json['vsaMetadata'].update({'capacityPackageName': ''})
        if self.parameters.get('capacity_package_name') is not None:
            json['vsaMetadata'].update({'capacityPackageName': self.parameters['capacity_package_name']})
        if self.parameters.get('project_id'):
            json.update({'project': self.parameters['project_id']})
        if self.parameters.get('nss_account'):
            json.update({'nssAccount': self.parameters['nss_account']})
        if self.parameters.get('subnet_id'):
            json.update({'subnetId': self.parameters['subnet_id']})
        if self.parameters.get('subnet_path'):
            json.update({'subnetPath': self.parameters['subnet_path']})
        if self.parameters.get('platform_serial_number') is not None:
            json.update({'serialNumber': self.parameters['platform_serial_number']})
        if self.parameters.get('capacity_tier') is not None and self.parameters['capacity_tier'] == 'cloudStorage':
            json.update({'capacityTier': self.parameters['capacity_tier'], 'tierLevel': self.parameters['tier_level']})
        if self.parameters.get('svm_name') is not None:
            json.update({'svmName': self.parameters['svm_name']})
        if self.parameters.get('gcp_labels') is not None:
            labels = []
            for each_label in self.parameters['gcp_labels']:
                label = {'labelKey': each_label['label_key'], 'labelValue': each_label['label_value']}
                labels.append(label)
            json.update({'gcpLabels': labels})
        if self.parameters.get('firewall_rule'):
            json.update({'firewallRule': self.parameters['firewall_rule']})
        if self.parameters['is_ha'] is True:
            ha_params = dict()
            if self.parameters.get('network_project_id') is not None:
                network_project_id = self.parameters.get('network_project_id')
            else:
                network_project_id = self.parameters['project_id']
            if not self.has_self_link(self.parameters['subnet_id']):
                json.update({'subnetId': 'projects/%s/regions/%s/subnetworks/%s' % (network_project_id, self.parameters['zone'][:-2], self.parameters['subnet_id'])})
            if self.parameters.get('platform_serial_number_node1'):
                ha_params['platformSerialNumberNode1'] = self.parameters['platform_serial_number_node1']
            if self.parameters.get('platform_serial_number_node2'):
                ha_params['platformSerialNumberNode2'] = self.parameters['platform_serial_number_node2']
            if self.parameters.get('node1_zone'):
                ha_params['node1Zone'] = self.parameters['node1_zone']
            if self.parameters.get('node2_zone'):
                ha_params['node2Zone'] = self.parameters['node2_zone']
            if self.parameters.get('mediator_zone'):
                ha_params['mediatorZone'] = self.parameters['mediator_zone']
            if self.parameters.get('vpc0_node_and_data_connectivity'):
                if self.has_self_link(self.parameters['vpc0_node_and_data_connectivity']):
                    ha_params['vpc0NodeAndDataConnectivity'] = self.parameters['vpc0_node_and_data_connectivity']
                else:
                    ha_params['vpc0NodeAndDataConnectivity'] = GOOGLE_API_URL + '/{0}/global/networks/{1}'.format(network_project_id, self.parameters['vpc0_node_and_data_connectivity'])
            if self.parameters.get('vpc1_cluster_connectivity'):
                if self.has_self_link(self.parameters['vpc1_cluster_connectivity']):
                    ha_params['vpc1ClusterConnectivity'] = self.parameters['vpc1_cluster_connectivity']
                else:
                    ha_params['vpc1ClusterConnectivity'] = GOOGLE_API_URL + '/{0}/global/networks/{1}'.format(network_project_id, self.parameters['vpc1_cluster_connectivity'])
            if self.parameters.get('vpc2_ha_connectivity'):
                if self.has_self_link(self.parameters['vpc2_ha_connectivity']):
                    ha_params['vpc2HAConnectivity'] = self.parameters['vpc2_ha_connectivity']
                else:
                    ha_params['vpc2HAConnectivity'] = 'https://www.googleapis.com/compute/v1/projects/{0}/global/networks/{1}'.format(network_project_id, self.parameters['vpc2_ha_connectivity'])
            if self.parameters.get('vpc3_data_replication'):
                if self.has_self_link(self.parameters['vpc3_data_replication']):
                    ha_params['vpc3DataReplication'] = self.parameters['vpc3_data_replication']
                else:
                    ha_params['vpc3DataReplication'] = GOOGLE_API_URL + '/{0}/global/networks/{1}'.format(network_project_id, self.parameters['vpc3_data_replication'])
            if self.parameters.get('subnet0_node_and_data_connectivity'):
                if self.has_self_link(self.parameters['subnet0_node_and_data_connectivity']):
                    ha_params['subnet0NodeAndDataConnectivity'] = self.parameters['subnet0_node_and_data_connectivity']
                else:
                    ha_params['subnet0NodeAndDataConnectivity'] = GOOGLE_API_URL + '/{0}/regions/{1}/subnetworks/{2}'.format(network_project_id, self.parameters['zone'][:-2], self.parameters['subnet0_node_and_data_connectivity'])
            if self.parameters.get('subnet1_cluster_connectivity'):
                if self.has_self_link(self.parameters['subnet1_cluster_connectivity']):
                    ha_params['subnet1ClusterConnectivity'] = self.parameters['subnet1_cluster_connectivity']
                else:
                    ha_params['subnet1ClusterConnectivity'] = GOOGLE_API_URL + '/{0}/regions/{1}/subnetworks/{2}'.format(network_project_id, self.parameters['zone'][:-2], self.parameters['subnet1_cluster_connectivity'])
            if self.parameters.get('subnet2_ha_connectivity'):
                if self.has_self_link(self.parameters['subnet2_ha_connectivity']):
                    ha_params['subnet2HAConnectivity'] = self.parameters['subnet2_ha_connectivity']
                else:
                    ha_params['subnet2HAConnectivity'] = GOOGLE_API_URL + '/{0}/regions/{1}/subnetworks/{2}'.format(network_project_id, self.parameters['zone'][:-2], self.parameters['subnet2_ha_connectivity'])
            if self.parameters.get('subnet3_data_replication'):
                if self.has_self_link(self.parameters['subnet3_data_replication']):
                    ha_params['subnet3DataReplication'] = self.parameters['subnet3_data_replication']
                else:
                    ha_params['subnet3DataReplication'] = GOOGLE_API_URL + '/{0}/regions/{1}/subnetworks/{2}'.format(network_project_id, self.parameters['zone'][:-2], self.parameters['subnet3_data_replication'])
            if self.parameters.get('vpc0_firewall_rule_name'):
                ha_params['vpc0FirewallRuleName'] = self.parameters['vpc0_firewall_ruleName']
            if self.parameters.get('vpc1_firewall_rule_name'):
                ha_params['vpc1FirewallRuleName'] = self.parameters['vpc1_firewall_rule_name']
            if self.parameters.get('vpc2_firewall_rule_name'):
                ha_params['vpc2FirewallRuleName'] = self.parameters['vpc2_firewall_rule_name']
            if self.parameters.get('vpc3_firewall_rule_name'):
                ha_params['vpc3FirewallRuleName'] = self.parameters['vpc3_firewall_rule_name']
            json['haParams'] = ha_params
        api_url = '%s/working-environments' % self.rest_api.api_root_path
        response, error, on_cloud_request_id = self.rest_api.post(api_url, json, header=self.headers)
        if error is not None:
            self.module.fail_json(msg='Error: unexpected response on creating cvo gcp: %s, %s' % (str(error), str(response)))
        working_environment_id = response['publicId']
        wait_on_completion_api_url = '/occm/api/audit/activeTask/%s' % str(on_cloud_request_id)
        err = self.rest_api.wait_on_completion(wait_on_completion_api_url, 'CVO', 'create', 60, 60)
        if err is not None:
            self.module.fail_json(msg='Error: unexpected response wait_on_completion for creating CVO GCP: %s' % str(err))
        return working_environment_id

    def update_cvo_gcp(self, working_environment_id, modify):
        base_url = '%s/working-environments/%s/' % (self.rest_api.api_root_path, working_environment_id)
        for item in modify:
            if item == 'svm_password':
                response, error = self.na_helper.update_svm_password(base_url, self.rest_api, self.headers, self.parameters['svm_password'])
                if error is not None:
                    self.module.fail_json(changed=False, msg=error)
            if item == 'svm_name':
                response, error = self.na_helper.update_svm_name(base_url, self.rest_api, self.headers, self.parameters['svm_name'])
                if error is not None:
                    self.module.fail_json(changed=False, msg=error)
            if item == 'gcp_labels':
                tag_list = None
                if 'gcp_labels' in self.parameters:
                    tag_list = self.parameters['gcp_labels']
                response, error = self.na_helper.update_cvo_tags(base_url, self.rest_api, self.headers, 'gcp_labels', tag_list)
                if error is not None:
                    self.module.fail_json(changed=False, msg=error)
            if item == 'tier_level':
                response, error = self.na_helper.update_tier_level(base_url, self.rest_api, self.headers, self.parameters['tier_level'])
                if error is not None:
                    self.module.fail_json(changed=False, msg=error)
            if item == 'writing_speed_state':
                response, error = self.na_helper.update_writing_speed_state(base_url, self.rest_api, self.headers, self.parameters['writing_speed_state'])
                if error is not None:
                    self.module.fail_json(changed=False, msg=error)
            if item == 'ontap_version':
                response, error = self.na_helper.upgrade_ontap_image(self.rest_api, self.headers, self.parameters['ontap_version'])
                if error is not None:
                    self.module.fail_json(changed=False, msg=error)
            if item == 'instance_type' or item == 'license_type':
                response, error = self.na_helper.update_instance_license_type(base_url, self.rest_api, self.headers, self.parameters['instance_type'], self.parameters['license_type'])
                if error is not None:
                    self.module.fail_json(changed=False, msg=error)

    def delete_cvo_gcp(self, we_id):
        """
        Delete GCP CVO
        """
        api_url = '%s/working-environments/%s' % (self.rest_api.api_root_path, we_id)
        response, error, on_cloud_request_id = self.rest_api.delete(api_url, None, header=self.headers)
        if error is not None:
            self.module.fail_json(msg='Error: unexpected response on deleting cvo gcp: %s, %s' % (str(error), str(response)))
        wait_on_completion_api_url = '/occm/api/audit/activeTask/%s' % str(on_cloud_request_id)
        err = self.rest_api.wait_on_completion(wait_on_completion_api_url, 'CVO', 'delete', 40, 60)
        if err is not None:
            self.module.fail_json(msg='Error: unexpected response wait_on_completion for deleting cvo gcp: %s' % str(err))

    def apply(self):
        working_environment_id = None
        modify = None
        current, dummy = self.na_helper.get_working_environment_details_by_name(self.rest_api, self.headers, self.parameters['name'], 'gcp')
        if current:
            self.parameters['working_environment_id'] = current['publicId']
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        if current and self.parameters['state'] != 'absent':
            working_environment_id = current['publicId']
            modify, error = self.na_helper.is_cvo_update_needed(self.rest_api, self.headers, self.parameters, self.changeable_params, 'gcp')
            if error is not None:
                self.module.fail_json(changed=False, msg=error)
        if self.na_helper.changed and (not self.module.check_mode):
            if cd_action == 'create':
                working_environment_id = self.create_cvo_gcp()
            elif cd_action == 'delete':
                self.delete_cvo_gcp(current['publicId'])
            else:
                self.update_cvo_gcp(current['publicId'], modify)
        self.module.exit_json(changed=self.na_helper.changed, working_environment_id=working_environment_id)