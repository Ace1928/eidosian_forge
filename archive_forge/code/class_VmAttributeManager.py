from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
class VmAttributeManager(PyVmomi):

    def __init__(self, module):
        super(VmAttributeManager, self).__init__(module)
        self.diff_config = dict(before={}, after={})
        self.result_fields = {}
        self.update_custom_attributes = []
        self.changed = False

    def set_custom_field(self, vm, user_fields):
        """Add or update the custom attribute and value.

        Args:
            vm (vim.VirtualMachine): The managed object of a virtual machine.
            user_fields (list): list of the specified custom attributes by user.

        Returns:
            The dictionary for the ansible return value.
        """
        self.check_exists(vm, user_fields)
        if self.module.check_mode is True:
            self.module.exit_json(changed=self.changed, diff=self.diff_config)
        for field in self.update_custom_attributes:
            if 'key' in field:
                self.content.customFieldsManager.SetField(entity=vm, key=field['key'], value=field['value'])
            else:
                field_key = self.content.customFieldsManager.AddFieldDefinition(name=field['name'], moType=vim.VirtualMachine)
                self.content.customFieldsManager.SetField(entity=vm, key=field_key.key, value=field['value'])
            self.result_fields[field['name']] = field['value']
        return {'changed': self.changed, 'failed': False, 'custom_attributes': self.result_fields}

    def remove_custom_field(self, vm, user_fields):
        """Remove the value from the existing custom attribute.

        Args:
            vm (vim.VirtualMachine): The managed object of a virtual machine.
            user_fields (list): list of the specified custom attributes by user.

        Returns:
            The dictionary for the ansible return value.
        """
        for v in user_fields:
            v['value'] = ''
        self.check_exists(vm, user_fields)
        if self.module.check_mode is True:
            self.module.exit_json(changed=self.changed, diff=self.diff_config)
        for field in self.update_custom_attributes:
            self.content.customFieldsManager.SetField(entity=vm, key=field['key'], value=field['value'])
            self.result_fields[field['name']] = field['value']
        return {'changed': self.changed, 'failed': False, 'custom_attributes': self.result_fields}

    def check_exists(self, vm, user_fields):
        """Check the existing custom attributes and values.

        In the function, the below processing is executed.

        Gather the existing custom attributes from the virtual machine and make update_custom_attributes for updating
        if it has differences between the existing configuration and the user_fields.

        And set diff key for checking between before and after configuration to self.diff_config.

        Args:
            vm (vim.VirtualMachine): The managed object of a virtual machine.
            user_fields (list): list of the specified custom attributes by user.
        """
        existing_custom_attributes = []
        for k, n in [(x.key, x.name) for x in self.custom_field_mgr if x.managedObjectType == vim.VirtualMachine or x.managedObjectType is None for v in user_fields if x.name == v['name']]:
            existing_custom_attributes.append({'key': k, 'name': n})
        for e in existing_custom_attributes:
            for v in vm.customValue:
                if e['key'] == v.key:
                    e['value'] = v.value
            if 'value' not in e:
                e['value'] = ''
        _user_fields_for_diff = []
        for v in user_fields:
            for e in existing_custom_attributes:
                if v['name'] == e['name'] and v['value'] != e['value']:
                    self.update_custom_attributes.append({'name': v['name'], 'value': v['value'], 'key': e['key']})
                if v['name'] == e['name']:
                    _user_fields_for_diff.append({'name': v['name'], 'value': v['value']})
            if v['name'] not in [x['name'] for x in existing_custom_attributes] and self.params['state'] == 'present':
                self.update_custom_attributes.append(v)
                _user_fields_for_diff.append({'name': v['name'], 'value': v['value']})
        if self.update_custom_attributes:
            self.changed = True
        self.diff_config['before']['custom_attributes'] = sorted([x for x in existing_custom_attributes if x.pop('key', None)], key=lambda k: k['name'])
        self.diff_config['after']['custom_attributes'] = sorted(_user_fields_for_diff, key=lambda k: k['name'])