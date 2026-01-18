from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import to_native
from ansible_collections.netapp.azure.plugins.module_utils.azure_rm_netapp_common import AzureRMNetAppModuleBase
from ansible_collections.netapp.azure.plugins.module_utils.netapp_module import NetAppModule
class AzureRMNetAppVolume(AzureRMNetAppModuleBase):
    """ create or delete a volume """

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), file_path=dict(type='str', required=False), pool_name=dict(type='str', required=True), account_name=dict(type='str', required=True), location=dict(type='str', required=False), state=dict(choices=['present', 'absent'], default='present', type='str'), subnet_name=dict(type='str', required=False, aliases=['subnet_id']), virtual_network=dict(type='str', required=False), size=dict(type='int', required=False), vnet_resource_group_for_subnet=dict(type='str', required=False), service_level=dict(type='str', required=False, choices=['Premium', 'Standard', 'Ultra']), protocol_types=dict(type='list', elements='str'), feature_flags=dict(type='dict'))
        self.na_helper = NetAppModule()
        self.parameters = {}
        super(AzureRMNetAppVolume, self).__init__(derived_arg_spec=self.module_arg_spec, required_if=[('state', 'present', ['location', 'file_path', 'subnet_name', 'virtual_network'])], supports_check_mode=True)

    @staticmethod
    def dict_from_volume_object(volume_object):

        def replace_list_of_objects_with_list_of_dicts(adict, key):
            if adict.get(key):
                adict[key] = [vars(x) for x in adict[key]]
        current_dict = vars(volume_object)
        attr = 'subnet_id'
        if attr in current_dict:
            current_dict['subnet_name'] = current_dict.pop(attr).split('/')[-1]
        attr = 'mount_targets'
        replace_list_of_objects_with_list_of_dicts(current_dict, attr)
        attr = 'export_policy'
        if current_dict.get(attr):
            attr_dict = vars(current_dict[attr])
            replace_list_of_objects_with_list_of_dicts(attr_dict, 'rules')
            current_dict[attr] = attr_dict
        return current_dict

    def get_azure_netapp_volume(self):
        """
            Returns volume object for an existing volume
            Return None if volume does not exist
        """
        try:
            volume_get = self.netapp_client.volumes.get(self.parameters['resource_group'], self.parameters['account_name'], self.parameters['pool_name'], self.parameters['name'])
        except (CloudError, ResourceNotFoundError):
            return None
        return self.dict_from_volume_object(volume_get)

    def get_export_policy_rules(self):
        ptypes = self.parameters.get('protocol_types')
        if ptypes is None:
            return None
        ptypes = [x.lower() for x in ptypes]
        if 'nfsv4.1' in ptypes:
            ptypes.append('nfsv41')
        if 'nfsv41' not in ptypes:
            return None
        options = dict(rule_index=1, allowed_clients='0.0.0.0/0', unix_read_write=True)
        if self.has_feature('ignore_change_ownership_mode') and self.sdk_version >= '4.0.0':
            options['chown_mode'] = None
        for protocol in ('cifs', 'nfsv3', 'nfsv41'):
            options[protocol] = protocol in ptypes
        return VolumePropertiesExportPolicy(rules=[ExportPolicyRule(**options)])

    def create_azure_netapp_volume(self):
        """
            Create a volume for the given Azure NetApp Account
            :return: None
        """
        options = self.na_helper.get_not_none_values_from_dict(self.parameters, ['protocol_types', 'service_level', 'tags', 'usage_threshold'])
        rules = self.get_export_policy_rules()
        if rules is not None:
            options['export_policy'] = rules
        subnet_id = '/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Network/virtualNetworks/%s/subnets/%s' % (self.azure_auth.subscription_id, self.parameters['resource_group'] if self.parameters.get('vnet_resource_group_for_subnet') is None else self.parameters['vnet_resource_group_for_subnet'], self.parameters['virtual_network'], self.parameters['subnet_name'])
        volume_body = Volume(location=self.parameters['location'], creation_token=self.parameters['file_path'], subnet_id=subnet_id, **options)
        try:
            result = self.get_method('volumes', 'create_or_update')(body=volume_body, resource_group_name=self.parameters['resource_group'], account_name=self.parameters['account_name'], pool_name=self.parameters['pool_name'], volume_name=self.parameters['name'])
            while result.done() is not True:
                result.result(10)
        except (CloudError, ValidationError, AzureError) as error:
            self.module.fail_json(msg='Error creating volume %s for Azure NetApp account %s and subnet ID %s: %s' % (self.parameters['name'], self.parameters['account_name'], subnet_id, to_native(error)), exception=traceback.format_exc())

    def modify_azure_netapp_volume(self):
        """
            Modify a volume for the given Azure NetApp Account
            :return: None
        """
        options = self.na_helper.get_not_none_values_from_dict(self.parameters, ['tags', 'usage_threshold'])
        volume_body = VolumePatch(**options)
        try:
            result = self.get_method('volumes', 'update')(body=volume_body, resource_group_name=self.parameters['resource_group'], account_name=self.parameters['account_name'], pool_name=self.parameters['pool_name'], volume_name=self.parameters['name'])
            while result.done() is not True:
                result.result(10)
        except (CloudError, ValidationError, AzureError) as error:
            self.module.fail_json(msg='Error modifying volume %s for Azure NetApp account %s: %s' % (self.parameters['name'], self.parameters['account_name'], to_native(error)), exception=traceback.format_exc())

    def delete_azure_netapp_volume(self):
        """
            Delete a volume for the given Azure NetApp Account
            :return: None
        """
        try:
            result = self.get_method('volumes', 'delete')(resource_group_name=self.parameters['resource_group'], account_name=self.parameters['account_name'], pool_name=self.parameters['pool_name'], volume_name=self.parameters['name'])
            while result.done() is not True:
                result.result(10)
        except (CloudError, AzureError) as error:
            self.module.fail_json(msg='Error deleting volume %s for Azure NetApp account %s: %s' % (self.parameters['name'], self.parameters['account_name'], to_native(error)), exception=traceback.format_exc())

    def validate_modify(self, modify, current):
        disallowed = dict(modify)
        disallowed.pop('tags', None)
        disallowed.pop('usage_threshold', None)
        if disallowed:
            self.module.fail_json(msg='Error: the following properties cannot be modified: %s.  Current: %s' % (repr(disallowed), repr(current)))

    def exec_module(self, **kwargs):
        self.fail_when_import_errors(IMPORT_ERRORS, HAS_AZURE_MGMT_NETAPP)
        for key in list(self.module_arg_spec):
            self.parameters[key] = kwargs[key]
        for key in ['tags']:
            if key in kwargs:
                self.parameters[key] = kwargs[key]
        if self.parameters.get('size') is not None:
            self.parameters['usage_threshold'] = ONE_GIB * self.parameters.pop('size')
        modify = None
        current = self.get_azure_netapp_volume()
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        if cd_action is None and current:
            name = current.pop('name', None)
            modify = self.na_helper.get_modified_attributes(current, self.parameters)
            if name is not None:
                current['name'] = name
            if 'tags' in modify:
                dummy, modify['tags'] = self.update_tags(current.get('tags'))
            self.validate_modify(modify, current)
        if self.na_helper.changed and (not self.module.check_mode):
            if cd_action == 'create':
                self.create_azure_netapp_volume()
            elif cd_action == 'delete':
                self.delete_azure_netapp_volume()
            elif modify:
                self.modify_azure_netapp_volume()

        def get_mount_info(return_info):
            if return_info is not None and return_info.get('mount_targets'):
                return '%s:/%s' % (return_info['mount_targets'][0]['ip_address'], return_info['creation_token'])
            return None
        mount_info = ''
        if self.parameters['state'] == 'present':
            return_info = self.get_azure_netapp_volume()
            if return_info is None and (not self.module.check_mode):
                self.module.fail_json(msg='Error: volume %s was created successfully, but cannot be found.' % self.parameters['name'])
            mount_info = get_mount_info(return_info)
            if mount_info is None and (not self.module.check_mode):
                self.module.fail_json(msg='Error: volume %s was created successfully, but mount target(s) cannot be found - volume details: %s.' % (self.parameters['name'], str(return_info)))
        self.module.exit_json(changed=self.na_helper.changed, mount_path=mount_info, modify=modify)