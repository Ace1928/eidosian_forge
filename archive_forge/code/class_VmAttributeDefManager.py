from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
class VmAttributeDefManager(PyVmomi):

    def __init__(self, module):
        super(VmAttributeDefManager, self).__init__(module)

    def remove_custom_def(self, field):
        changed = False
        f = dict()
        for x in self.custom_field_mgr:
            if x.name == field and x.managedObjectType == vim.VirtualMachine:
                changed = True
                if not self.module.check_mode:
                    self.content.customFieldsManager.RemoveCustomFieldDef(key=x.key)
                    break
            f[x.name] = (x.key, x.managedObjectType)
        return {'changed': changed, 'failed': False, 'custom_attribute_defs': list(f.keys())}

    def add_custom_def(self, field):
        changed = False
        found = False
        f = dict()
        for x in self.custom_field_mgr:
            if x.name == field:
                found = True
            f[x.name] = (x.key, x.managedObjectType)
        if not found:
            changed = True
            if not self.module.check_mode:
                new_field = self.content.customFieldsManager.AddFieldDefinition(name=field, moType=vim.VirtualMachine)
                f[new_field.name] = (new_field.key, new_field.type)
        return {'changed': changed, 'failed': False, 'custom_attribute_defs': list(f.keys())}