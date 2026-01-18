from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.meraki.plugins.plugin_utils.meraki import (
from ansible_collections.cisco.meraki.plugins.plugin_utils.exceptions import (
class NetworksApplianceVpnBgp(object):

    def __init__(self, params, meraki):
        self.meraki = meraki
        self.new_object = dict(enabled=params.get('enabled'), asNumber=params.get('asNumber'), ibgpHoldTimer=params.get('ibgpHoldTimer'), neighbors=params.get('neighbors'), network_id=params.get('networkId'))

    def get_all_params(self, name=None, id=None):
        new_object_params = {}
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        return new_object_params

    def update_all_params(self):
        new_object_params = {}
        if self.new_object.get('enabled') is not None or self.new_object.get('enabled') is not None:
            new_object_params['enabled'] = self.new_object.get('enabled')
        if self.new_object.get('asNumber') is not None or self.new_object.get('as_number') is not None:
            new_object_params['asNumber'] = self.new_object.get('asNumber') or self.new_object.get('as_number')
        if self.new_object.get('ibgpHoldTimer') is not None or self.new_object.get('ibgp_hold_timer') is not None:
            new_object_params['ibgpHoldTimer'] = self.new_object.get('ibgpHoldTimer') or self.new_object.get('ibgp_hold_timer')
        if self.new_object.get('neighbors') is not None or self.new_object.get('neighbors') is not None:
            new_object_params['neighbors'] = self.new_object.get('neighbors') or self.new_object.get('neighbors')
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        return new_object_params

    def get_object_by_name(self, name):
        result = None
        try:
            items = self.meraki.exec_meraki(family='appliance', function='getNetworkApplianceVpnBgp', params=self.get_all_params(name=name))
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
        obj_params = [('enabled', 'enabled'), ('asNumber', 'asNumber'), ('ibgpHoldTimer', 'ibgpHoldTimer'), ('neighbors', 'neighbors'), ('networkId', 'networkId')]
        return any((not meraki_compare_equality(current_obj.get(meraki_param), requested_obj.get(ansible_param)) for meraki_param, ansible_param in obj_params))

    def update(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        result = self.meraki.exec_meraki(family='appliance', function='updateNetworkApplianceVpnBgp', params=self.update_all_params(), op_modifies=True)
        return result