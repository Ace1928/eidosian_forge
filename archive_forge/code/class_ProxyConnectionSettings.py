from __future__ import absolute_import, division, print_function
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.ise.plugins.plugin_utils.ise import (
from ansible_collections.cisco.ise.plugins.plugin_utils.exceptions import (
class ProxyConnectionSettings(object):

    def __init__(self, params, ise):
        self.ise = ise
        self.new_object = dict(bypass_hosts=params.get('bypassHosts'), fqdn=params.get('fqdn'), password=params.get('password'), password_required=params.get('passwordRequired'), port=params.get('port'), user_name=params.get('userName'))

    def get_object_by_name(self, name):
        result = None
        items = self.ise.exec(family='proxy', function='get_proxy_connection').response['response']
        result = get_dict_result(items, 'name', name)
        return result

    def get_object_by_id(self, id):
        result = None
        return result

    def exists(self):
        prev_obj = None
        id_exists = False
        name_exists = False
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
        it_exists = prev_obj is not None and isinstance(prev_obj, dict)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('bypassHosts', 'bypass_hosts'), ('fqdn', 'fqdn'), ('password', 'password'), ('passwordRequired', 'password_required'), ('port', 'port'), ('userName', 'user_name')]
        return any((not ise_compare_equality(current_obj.get(ise_param), requested_obj.get(ansible_param)) for ise_param, ansible_param in obj_params))

    def update(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        result = self.ise.exec(family='proxy', function='update_proxy_connection', params=self.new_object).response
        return result