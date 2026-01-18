from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.meraki.plugins.plugin_utils.meraki import (
from ansible_collections.cisco.meraki.plugins.plugin_utils.exceptions import (
class NetworksSmBypassActivationLockAttempts(object):

    def __init__(self, params, meraki):
        self.meraki = meraki
        self.new_object = dict(ids=params.get('ids'), network_id=params.get('networkId'), attempt_id=params.get('attemptId'))

    def get_params_by_id(self, name=None, id=None):
        new_object_params = {}
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        if self.new_object.get('attemptId') is not None or self.new_object.get('attempt_id') is not None:
            new_object_params['attemptId'] = self.new_object.get('attemptId') or self.new_object.get('attempt_id')
        return new_object_params

    def create_params(self):
        new_object_params = {}
        if self.new_object.get('ids') is not None or self.new_object.get('ids') is not None:
            new_object_params['ids'] = self.new_object.get('ids') or self.new_object.get('ids')
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        return new_object_params

    def get_object_by_name(self, name):
        result = None
        return result

    def get_object_by_id(self, id):
        result = None
        try:
            items = self.meraki.exec_meraki(family='sm', function='getNetworkSmBypassActivationLockAttempt', params=self.get_params_by_id())
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
            result = items
        except Exception as e:
            print('Error: ', e)
            result = None
        return result

    def exists(self):
        prev_obj = None
        id_exists = False
        name_exists = False
        o_id = self.new_object.get('networkId') or self.new_object.get('network_id')
        o_id = o_id or self.new_object.get('attempt_id') or self.new_object.get('attemptId')
        name = self.new_object.get('name')
        if o_id:
            prev_obj = self.get_object_by_id(o_id)
            id_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if not id_exists and name:
            prev_obj = self.get_object_by_name(name)
            name_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if name_exists:
            _id = prev_obj.get('id')
            _id = _id or prev_obj.get('attemptId')
            if id_exists and name_exists and (o_id != _id):
                raise InconsistentParameters("The 'id' and 'name' params don't refer to the same object")
            if _id:
                self.new_object.update(dict(id=_id))
                self.new_object.update(dict(attemptId=_id))
            if _id:
                prev_obj = self.get_object_by_id(_id)
        it_exists = prev_obj is not None and isinstance(prev_obj, dict)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('ids', 'ids'), ('networkId', 'networkId'), ('attemptId', 'attemptId')]
        return any((not meraki_compare_equality(current_obj.get(meraki_param), requested_obj.get(ansible_param)) for meraki_param, ansible_param in obj_params))

    def create(self):
        result = self.meraki.exec_meraki(family='sm', function='createNetworkSmBypassActivationLockAttempt', params=self.create_params(), op_modifies=True)
        return result