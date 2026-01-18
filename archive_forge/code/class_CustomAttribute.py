from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
class CustomAttribute(PyVmomi):

    def __init__(self, module):
        super(CustomAttribute, self).__init__(module)
        if not self.is_vcenter():
            self.module.fail_json(msg='You have to connect to a vCenter server!')
        object_types_map = {'Cluster': vim.ClusterComputeResource, 'Datacenter': vim.Datacenter, 'Datastore': vim.Datastore, 'DistributedVirtualPortgroup': vim.DistributedVirtualPortgroup, 'DistributedVirtualSwitch': vim.DistributedVirtualSwitch, 'Folder': vim.Folder, 'Global': None, 'HostSystem': vim.HostSystem, 'ResourcePool': vim.ResourcePool, 'VirtualMachine': vim.VirtualMachine}
        self.object_type = object_types_map[self.params['object_type']]

    def remove_custom_def(self, field):
        changed = False
        for x in self.custom_field_mgr:
            if x.name == field and x.managedObjectType == self.object_type:
                changed = True
                if not self.module.check_mode:
                    self.content.customFieldsManager.RemoveCustomFieldDef(key=x.key)
                break
        return {'changed': changed, 'failed': False}

    def add_custom_def(self, field):
        changed = False
        found = False
        for x in self.custom_field_mgr:
            if x.name == field and x.managedObjectType == self.object_type:
                found = True
                break
        if not found:
            changed = True
            if not self.module.check_mode:
                self.content.customFieldsManager.AddFieldDefinition(name=field, moType=self.object_type)
        return {'changed': changed, 'failed': False}