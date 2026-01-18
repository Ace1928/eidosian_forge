from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.dnac.plugins.plugin_utils.dnac import (
from ansible_collections.cisco.dnac.plugins.plugin_utils.exceptions import (
class NetworkDeviceUserDefinedField(object):

    def __init__(self, params, dnac):
        self.dnac = dnac
        self.new_object = dict(name=params.get('name'), description=params.get('description'), id=params.get('id'))

    def get_all_params(self, name=None, id=None):
        new_object_params = {}
        new_object_params['id'] = id or self.new_object.get('id')
        new_object_params['name'] = name or self.new_object.get('name')
        return new_object_params

    def create_params(self):
        new_object_params = {}
        new_object_params['name'] = self.new_object.get('name')
        new_object_params['description'] = self.new_object.get('description')
        return new_object_params

    def delete_by_id_params(self):
        new_object_params = {}
        new_object_params['id'] = self.new_object.get('id')
        return new_object_params

    def update_by_id_params(self):
        new_object_params = {}
        new_object_params['name'] = self.new_object.get('name')
        new_object_params['description'] = self.new_object.get('description')
        return new_object_params

    def get_object_by_name(self, name):
        result = None
        try:
            items = self.dnac.exec(family='devices', function='get_all_user_defined_fields', params=self.get_all_params(name=name))
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
            result = get_dict_result(items, 'name', name)
        except Exception:
            result = None
        return result

    def get_object_by_id(self, id):
        result = None
        try:
            items = self.dnac.exec(family='devices', function='get_all_user_defined_fields', params=self.get_all_params(id=id))
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
            result = get_dict_result(items, 'id', id)
        except Exception:
            result = None
        return result

    def exists(self):
        id_exists = False
        name_exists = False
        prev_obj = None
        o_id = self.new_object.get('id')
        name = self.new_object.get('name')
        if o_id:
            prev_obj = self.get_object_by_id(o_id)
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
        obj_params = [('name', 'name'), ('description', 'description'), ('id', 'id')]
        return any((not dnac_compare_equality(current_obj.get(dnac_param), requested_obj.get(ansible_param)) for dnac_param, ansible_param in obj_params))

    def create(self):
        result = self.dnac.exec(family='devices', function='create_user_defined_field', params=self.create_params(), op_modifies=True)
        return result

    def update(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
            if id_:
                self.new_object.update(dict(id=id_))
        result = self.dnac.exec(family='devices', function='update_user_defined_field', params=self.update_by_id_params(), op_modifies=True)
        return result

    def delete(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
            if id_:
                self.new_object.update(dict(id=id_))
        result = self.dnac.exec(family='devices', function='delete_user_defined_field', params=self.delete_by_id_params())
        return result