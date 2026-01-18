from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.validation import (
from ansible_collections.community.general.plugins.module_utils.redfish_utils import RedfishUtils
from ansible.module_utils.common.text.converters import to_native
class IdracRedfishUtils(RedfishUtils):

    def set_manager_attributes(self, command):
        result = {}
        required_arg_spec = {'manager_attributes': {'required': True}}
        try:
            check_required_arguments(required_arg_spec, self.module.params)
        except TypeError as e:
            msg = to_native(e)
            self.module.fail_json(msg=msg)
        key = 'Attributes'
        command_manager_attributes_uri_map = {'SetManagerAttributes': self.manager_uri, 'SetLifecycleControllerAttributes': '/redfish/v1/Managers/LifecycleController.Embedded.1', 'SetSystemAttributes': '/redfish/v1/Managers/System.Embedded.1'}
        manager_uri = command_manager_attributes_uri_map.get(command, self.manager_uri)
        attributes = self.module.params['manager_attributes']
        attrs_to_patch = {}
        attrs_skipped = {}
        attrs_bad = {}
        response = self.get_request(self.root_uri + manager_uri + '/' + key)
        if response['ret'] is False:
            return response
        result['ret'] = True
        data = response['data']
        if key not in data:
            return {'ret': False, 'msg': '%s: Key %s not found' % (command, key), 'warning': ''}
        for attr_name, attr_value in attributes.items():
            if attr_name not in data[u'Attributes']:
                attrs_bad.update({attr_name: attr_value})
                continue
            if data[u'Attributes'][attr_name] == attr_value:
                attrs_skipped.update({attr_name: attr_value})
            else:
                attrs_to_patch.update({attr_name: attr_value})
        warning = ''
        if attrs_bad:
            warning = 'Incorrect attributes %s' % attrs_bad
        if not attrs_to_patch:
            return {'ret': True, 'changed': False, 'msg': 'No changes made. Manager attributes already set.', 'warning': warning}
        payload = {'Attributes': attrs_to_patch}
        response = self.patch_request(self.root_uri + manager_uri + '/' + key, payload)
        if response['ret'] is False:
            return response
        return {'ret': True, 'changed': True, 'msg': '%s: Modified Manager attributes %s' % (command, attrs_to_patch), 'warning': warning}