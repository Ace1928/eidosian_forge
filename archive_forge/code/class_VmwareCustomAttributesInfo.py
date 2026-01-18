from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, find_obj
class VmwareCustomAttributesInfo(PyVmomi):

    def __init__(self, module):
        super(VmwareCustomAttributesInfo, self).__init__(module)
        if not self.is_vcenter():
            self.module.fail_json(msg='You have to connect to a vCenter server!')
        self.object_type = self.params['object_type']
        self.object_name = self.params['object_name']
        self.moid = self.params['moid']
        self.valid_object_types = {'Datacenter': vim.Datacenter, 'Cluster': vim.ClusterComputeResource, 'HostSystem': vim.HostSystem, 'ResourcePool': vim.ResourcePool, 'Folder': vim.Folder, 'VirtualMachine': vim.VirtualMachine, 'DistributedVirtualSwitch': vim.DistributedVirtualSwitch, 'DistributedVirtualPortgroup': vim.DistributedVirtualPortgroup, 'Datastore': vim.Datastore}

    def execute(self):
        result = {'changed': False}
        if self.object_name:
            obj = find_obj(self.content, [self.valid_object_types[self.object_type]], self.object_name)
        elif self.moid:
            obj = self.find_obj_by_moid(self.object_type, self.moid)
        if not obj:
            self.module.fail_json(msg="can't find the object: %s" % self.object_name if self.object_name else self.moid)
        custom_attributes = []
        available_fields = {}
        for available_custom_attribute in obj.availableField:
            available_fields.update({available_custom_attribute.key: {'name': available_custom_attribute.name, 'type': available_custom_attribute.managedObjectType}})
        custom_values = {}
        for custom_value in obj.customValue:
            custom_values.update({custom_value.key: custom_value.value})
        for key, value in available_fields.items():
            attribute_result = {'attribute': value['name'], 'type': self.to_json(value['type']).replace('vim.', '') if value['type'] is not None else 'Global', 'key': key, 'value': None}
            if key in custom_values:
                attribute_result['value'] = custom_values[key]
            custom_attributes.append(attribute_result)
        result['custom_attributes'] = custom_attributes
        self.module.exit_json(**result)