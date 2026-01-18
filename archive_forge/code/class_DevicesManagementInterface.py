from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.meraki.plugins.plugin_utils.meraki import (
from ansible_collections.cisco.meraki.plugins.plugin_utils.exceptions import (
class DevicesManagementInterface(object):

    def __init__(self, params, meraki):
        self.meraki = meraki
        self.new_object = dict(serial=params.get('serial'), wan1=params.get('wan1'), wan2=params.get('wan2'))

    def get_all_params(self, name=None, id=None):
        new_object_params = {}
        if self.new_object.get('serial') is not None or self.new_object.get('serial') is not None:
            new_object_params['serial'] = self.new_object.get('serial')
        return new_object_params

    def create_params(self):
        new_object_params = {}
        if self.new_object.get('serial') is not None or self.new_object.get('serial') is not None:
            new_object_params['serial'] = self.new_object.get('serial') or self.new_object.get('serial')
        return new_object_params

    def update_all_params(self):
        new_object_params = {}
        if self.new_object.get('wan1') is not None or self.new_object.get('wan1') is not None:
            new_object_params['wan1'] = self.new_object.get('wan1') or self.new_object.get('wan1')
        if self.new_object.get('wan2') is not None or self.new_object.get('wan2') is not None:
            new_object_params['wan2'] = self.new_object.get('wan2') or self.new_object.get('wan2')
        if self.new_object.get('serial') is not None or self.new_object.get('serial') is not None:
            new_object_params['serial'] = self.new_object.get('serial') or self.new_object.get('serial')
        return new_object_params

    def get_object_by_name(self, name):
        result = None
        try:
            items = self.meraki.exec_meraki(family='devices', function='getDeviceManagementInterface', params=self.get_all_params(name=name))
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
            result = get_dict_result(items, 'name', name)
            if result is None:
                result = items
        except Exception as e:
            print('Error: ', e)
            result = None
        return result

    def get_object_by_id(self, id):
        result = None
        return result

    def exists(self):
        prev_obj = None
        id_exists = False
        name_exists = False
        o_id = self.new_object.get('serial')
        name = self.new_object.get('name')
        if o_id:
            prev_obj = self.get_object_by_name(o_id)
            id_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if not id_exists and name:
            prev_obj = self.get_object_by_name(name)
            name_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if name_exists:
            _id = prev_obj.get('id')
            if id_exists and name_exists and (o_id != _id):
                raise InconsistentParameters("The 'id' and 'name' params don't refer to the same object")
            if _id:
                self.new_object.update(dict(id=_id))
        it_exists = prev_obj is not None and isinstance(prev_obj, dict)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('serial', 'serial'), ('wan1', 'wan1'), ('wan2', 'wan2')]
        return any((not meraki_compare_equality(current_obj.get(meraki_param), requested_obj.get(ansible_param)) for meraki_param, ansible_param in obj_params))

    def create(self):
        result = self.meraki.exec_meraki(family='devices', function='rebootDevice', params=self.create_params(), op_modifies=True)
        return result

    def update(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        result = self.meraki.exec_meraki(family='devices', function='updateDeviceManagementInterface', params=self.update_all_params(), op_modifies=True)
        return result