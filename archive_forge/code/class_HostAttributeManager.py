from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
class HostAttributeManager(PyVmomi):

    def __init__(self, module):
        super(HostAttributeManager, self).__init__(module)
        self.esxi_hostname = module.params.get('esxi_hostname')
        self.host = self.find_hostsystem_by_name(self.esxi_hostname)

    def set_custom_field(self, host, user_fields):
        result_fields = dict()
        change_list = list()
        changed = False
        for field in user_fields:
            field_key = self.check_exists(field['name'])
            found = False
            field_value = field.get('value', '')
            for k, v in [(x.name, v.value) for x in self.custom_field_mgr for v in host.customValue if x.key == v.key]:
                if k == field['name']:
                    found = True
                    if v != field_value:
                        if not self.module.check_mode:
                            self.content.customFieldsManager.SetField(entity=host, key=field_key.key, value=field_value)
                            result_fields[k] = field_value
                        change_list.append(True)
            if not found and field_value != '':
                if not field_key and (not self.module.check_mode):
                    field_key = self.content.customFieldsManager.AddFieldDefinition(name=field['name'], moType=vim.HostSystem)
                change_list.append(True)
                if not self.module.check_mode:
                    self.content.customFieldsManager.SetField(entity=host, key=field_key.key, value=field_value)
                result_fields[field['name']] = field_value
        if any(change_list):
            changed = True
        return {'changed': changed, 'failed': False, 'custom_attributes': result_fields}

    def check_exists(self, field):
        for x in self.custom_field_mgr:
            if x.managedObjectType in (None, vim.HostSystem) and x.name == field:
                return x
        return False