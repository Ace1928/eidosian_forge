from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.meraki.plugins.plugin_utils.meraki import (
from ansible_collections.cisco.meraki.plugins.plugin_utils.exceptions import (
class NetworksSwitchRoutingOspf(object):

    def __init__(self, params, meraki):
        self.meraki = meraki
        self.new_object = dict(enabled=params.get('enabled'), helloTimerInSeconds=params.get('helloTimerInSeconds'), deadTimerInSeconds=params.get('deadTimerInSeconds'), areas=params.get('areas'), v3=params.get('v3'), md5AuthenticationEnabled=params.get('md5AuthenticationEnabled'), md5AuthenticationKey=params.get('md5AuthenticationKey'), network_id=params.get('networkId'))

    def get_all_params(self, name=None, id=None):
        new_object_params = {}
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        return new_object_params

    def update_all_params(self):
        new_object_params = {}
        if self.new_object.get('enabled') is not None or self.new_object.get('enabled') is not None:
            new_object_params['enabled'] = self.new_object.get('enabled')
        if self.new_object.get('helloTimerInSeconds') is not None or self.new_object.get('hello_timer_in_seconds') is not None:
            new_object_params['helloTimerInSeconds'] = self.new_object.get('helloTimerInSeconds') or self.new_object.get('hello_timer_in_seconds')
        if self.new_object.get('deadTimerInSeconds') is not None or self.new_object.get('dead_timer_in_seconds') is not None:
            new_object_params['deadTimerInSeconds'] = self.new_object.get('deadTimerInSeconds') or self.new_object.get('dead_timer_in_seconds')
        if self.new_object.get('areas') is not None or self.new_object.get('areas') is not None:
            new_object_params['areas'] = self.new_object.get('areas') or self.new_object.get('areas')
        if self.new_object.get('v3') is not None or self.new_object.get('v3') is not None:
            new_object_params['v3'] = self.new_object.get('v3') or self.new_object.get('v3')
        if self.new_object.get('md5AuthenticationEnabled') is not None or self.new_object.get('md5_authentication_enabled') is not None:
            new_object_params['md5AuthenticationEnabled'] = self.new_object.get('md5AuthenticationEnabled')
        if self.new_object.get('md5AuthenticationKey') is not None or self.new_object.get('md5_authentication_key') is not None:
            new_object_params['md5AuthenticationKey'] = self.new_object.get('md5AuthenticationKey') or self.new_object.get('md5_authentication_key')
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        return new_object_params

    def get_object_by_name(self, name):
        result = None
        try:
            items = self.meraki.exec_meraki(family='switch', function='getNetworkSwitchRoutingOspf', params=self.get_all_params(name=name))
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
        o_id = self.new_object.get('networkId') or self.new_object.get('network_id')
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
        obj_params = [('enabled', 'enabled'), ('helloTimerInSeconds', 'helloTimerInSeconds'), ('deadTimerInSeconds', 'deadTimerInSeconds'), ('areas', 'areas'), ('v3', 'v3'), ('md5AuthenticationEnabled', 'md5AuthenticationEnabled'), ('md5AuthenticationKey', 'md5AuthenticationKey'), ('networkId', 'networkId')]
        return any((not meraki_compare_equality(current_obj.get(meraki_param), requested_obj.get(ansible_param)) for meraki_param, ansible_param in obj_params))

    def update(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        result = self.meraki.exec_meraki(family='switch', function='updateNetworkSwitchRoutingOspf', params=self.update_all_params(), op_modifies=True)
        return result