from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.dnac.plugins.plugin_utils.dnac import (
from ansible_collections.cisco.dnac.plugins.plugin_utils.exceptions import (
class HttpWriteCredential(object):

    def __init__(self, params, dnac):
        self.dnac = dnac
        self.new_object = dict(comments=params.get('comments'), credentialType=params.get('credentialType'), description=params.get('description'), id=params.get('id'), instanceTenantId=params.get('instanceTenantId'), instanceUuid=params.get('instanceUuid'), password=params.get('password'), port=params.get('port'), secure=params.get('secure'), username=params.get('username'))

    def create_params(self):
        new_object_params = {}
        payload = {}
        keys = ['comments', 'credentialType', 'description', 'id', 'instanceTenantId', 'instanceUuid', 'password', 'port', 'secure', 'username']
        for key in keys:
            if self.new_object.get(key) is not None:
                payload[key] = self.new_object.get(key)
        new_object_params['payload'] = [payload]
        return new_object_params

    def update_all_params(self):
        new_object_params = {}
        new_object_params['comments'] = self.new_object.get('comments')
        new_object_params['credentialType'] = self.new_object.get('credentialType')
        new_object_params['description'] = self.new_object.get('description')
        new_object_params['id'] = self.new_object.get('id')
        new_object_params['instanceTenantId'] = self.new_object.get('instanceTenantId')
        new_object_params['instanceUuid'] = self.new_object.get('instanceUuid')
        new_object_params['password'] = self.new_object.get('password')
        new_object_params['port'] = self.new_object.get('port')
        new_object_params['secure'] = self.new_object.get('secure')
        new_object_params['username'] = self.new_object.get('username')
        return new_object_params

    def get_object_by_name(self, name):
        result = None
        try:
            items = self.dnac.exec(family='discovery', function='get_global_credentials', params={'credential_sub_type': 'HTTP_WRITE'})
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
            result = get_dict_result(items, 'description', name) or get_dict_result(items, 'username', name)
        except Exception:
            result = None
        return result

    def get_object_by_id(self, id):
        result = None
        try:
            items = self.dnac.exec(family='discovery', function='get_global_credentials', params={'credential_sub_type': 'HTTP_WRITE'})
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
            result = get_dict_result(items, 'id', id)
        except Exception:
            result = None
        return result

    def exists(self):
        prev_obj = None
        id_exists = False
        name_exists = False
        o_id = self.new_object.get('id')
        name = self.new_object.get('username') or self.new_object.get('description')
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
        obj_params = [('comments', 'comments'), ('credentialType', 'credentialType'), ('description', 'description'), ('id', 'id'), ('instanceTenantId', 'instanceTenantId'), ('instanceUuid', 'instanceUuid'), ('port', 'port'), ('secure', 'secure'), ('username', 'username')]
        return any((not dnac_compare_equality(current_obj.get(dnac_param), requested_obj.get(ansible_param)) for dnac_param, ansible_param in obj_params))

    def create(self):
        result = self.dnac.exec(family='discovery', function='create_http_write_credentials', params=self.create_params(), op_modifies=True)
        return result

    def update(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        result = self.dnac.exec(family='discovery', function='update_http_write_credentials', params=self.update_all_params(), op_modifies=True)
        return result