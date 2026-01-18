from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.meraki.plugins.plugin_utils.meraki import (
from ansible_collections.cisco.meraki.plugins.plugin_utils.exceptions import (
class NetworksSwitchAccessPolicies(object):

    def __init__(self, params, meraki):
        self.meraki = meraki
        self.new_object = dict(name=params.get('name'), radiusServers=params.get('radiusServers'), radius=params.get('radius'), guestPortBouncing=params.get('guestPortBouncing'), radiusTestingEnabled=params.get('radiusTestingEnabled'), radiusCoaSupportEnabled=params.get('radiusCoaSupportEnabled'), radiusAccountingEnabled=params.get('radiusAccountingEnabled'), radiusAccountingServers=params.get('radiusAccountingServers'), radiusGroupAttribute=params.get('radiusGroupAttribute'), hostMode=params.get('hostMode'), accessPolicyType=params.get('accessPolicyType'), increaseAccessSpeed=params.get('increaseAccessSpeed'), guestVlanId=params.get('guestVlanId'), dot1x=params.get('dot1x'), voiceVlanClients=params.get('voiceVlanClients'), urlRedirectWalledGardenEnabled=params.get('urlRedirectWalledGardenEnabled'), urlRedirectWalledGardenRanges=params.get('urlRedirectWalledGardenRanges'), networkId=params.get('networkId'), accessPolicyNumber=params.get('accessPolicyNumber'))

    def get_all_params(self, name=None, id=None):
        new_object_params = {}
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        return new_object_params

    def get_params_by_id(self, name=None, id=None):
        new_object_params = {}
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        if self.new_object.get('accessPolicyNumber') is not None or self.new_object.get('access_policy_number') is not None:
            new_object_params['accessPolicyNumber'] = self.new_object.get('accessPolicyNumber') or self.new_object.get('access_policy_number')
        return new_object_params

    def create_params(self):
        new_object_params = {}
        if self.new_object.get('name') is not None or self.new_object.get('name') is not None:
            new_object_params['name'] = self.new_object.get('name') or self.new_object.get('name')
        if self.new_object.get('radiusServers') is not None or self.new_object.get('radius_servers') is not None:
            new_object_params['radiusServers'] = self.new_object.get('radiusServers') or self.new_object.get('radius_servers')
        if self.new_object.get('radius') is not None or self.new_object.get('radius') is not None:
            new_object_params['radius'] = self.new_object.get('radius') or self.new_object.get('radius')
        if self.new_object.get('guestPortBouncing') is not None or self.new_object.get('guest_port_bouncing') is not None:
            new_object_params['guestPortBouncing'] = self.new_object.get('guestPortBouncing')
        if self.new_object.get('radiusTestingEnabled') is not None or self.new_object.get('radius_testing_enabled') is not None:
            new_object_params['radiusTestingEnabled'] = self.new_object.get('radiusTestingEnabled')
        if self.new_object.get('radiusCoaSupportEnabled') is not None or self.new_object.get('radius_coa_support_enabled') is not None:
            new_object_params['radiusCoaSupportEnabled'] = self.new_object.get('radiusCoaSupportEnabled')
        if self.new_object.get('radiusAccountingEnabled') is not None or self.new_object.get('radius_accounting_enabled') is not None:
            new_object_params['radiusAccountingEnabled'] = self.new_object.get('radiusAccountingEnabled')
        if self.new_object.get('radiusAccountingServers') is not None or self.new_object.get('radius_accounting_servers') is not None:
            new_object_params['radiusAccountingServers'] = self.new_object.get('radiusAccountingServers') or self.new_object.get('radius_accounting_servers')
        if self.new_object.get('radiusGroupAttribute') is not None or self.new_object.get('radius_group_attribute') is not None:
            new_object_params['radiusGroupAttribute'] = self.new_object.get('radiusGroupAttribute') or self.new_object.get('radius_group_attribute')
        if self.new_object.get('hostMode') is not None or self.new_object.get('host_mode') is not None:
            new_object_params['hostMode'] = self.new_object.get('hostMode') or self.new_object.get('host_mode')
        if self.new_object.get('accessPolicyType') is not None or self.new_object.get('access_policy_type') is not None:
            new_object_params['accessPolicyType'] = self.new_object.get('accessPolicyType') or self.new_object.get('access_policy_type')
        if self.new_object.get('increaseAccessSpeed') is not None or self.new_object.get('increase_access_speed') is not None:
            new_object_params['increaseAccessSpeed'] = self.new_object.get('increaseAccessSpeed')
        if self.new_object.get('guestVlanId') is not None or self.new_object.get('guest_vlan_id') is not None:
            new_object_params['guestVlanId'] = self.new_object.get('guestVlanId') or self.new_object.get('guest_vlan_id')
        if self.new_object.get('dot1x') is not None or self.new_object.get('dot1x') is not None:
            new_object_params['dot1x'] = self.new_object.get('dot1x') or self.new_object.get('dot1x')
        if self.new_object.get('voiceVlanClients') is not None or self.new_object.get('voice_vlan_clients') is not None:
            new_object_params['voiceVlanClients'] = self.new_object.get('voiceVlanClients')
        if self.new_object.get('urlRedirectWalledGardenEnabled') is not None or self.new_object.get('url_redirect_walled_garden_enabled') is not None:
            new_object_params['urlRedirectWalledGardenEnabled'] = self.new_object.get('urlRedirectWalledGardenEnabled')
        if self.new_object.get('urlRedirectWalledGardenRanges') is not None or self.new_object.get('url_redirect_walled_garden_ranges') is not None:
            new_object_params['urlRedirectWalledGardenRanges'] = self.new_object.get('urlRedirectWalledGardenRanges') or self.new_object.get('url_redirect_walled_garden_ranges')
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        return new_object_params

    def delete_by_id_params(self):
        new_object_params = {}
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        if self.new_object.get('accessPolicyNumber') is not None or self.new_object.get('access_policy_number') is not None:
            new_object_params['accessPolicyNumber'] = self.new_object.get('accessPolicyNumber') or self.new_object.get('access_policy_number')
        return new_object_params

    def update_by_id_params(self):
        new_object_params = {}
        if self.new_object.get('name') is not None or self.new_object.get('name') is not None:
            new_object_params['name'] = self.new_object.get('name') or self.new_object.get('name')
        if self.new_object.get('radiusServers') is not None or self.new_object.get('radius_servers') is not None:
            new_object_params['radiusServers'] = self.new_object.get('radiusServers') or self.new_object.get('radius_servers')
        if self.new_object.get('radius') is not None or self.new_object.get('radius') is not None:
            new_object_params['radius'] = self.new_object.get('radius') or self.new_object.get('radius')
        if self.new_object.get('guestPortBouncing') is not None or self.new_object.get('guest_port_bouncing') is not None:
            new_object_params['guestPortBouncing'] = self.new_object.get('guestPortBouncing')
        if self.new_object.get('radiusTestingEnabled') is not None or self.new_object.get('radius_testing_enabled') is not None:
            new_object_params['radiusTestingEnabled'] = self.new_object.get('radiusTestingEnabled')
        if self.new_object.get('radiusCoaSupportEnabled') is not None or self.new_object.get('radius_coa_support_enabled') is not None:
            new_object_params['radiusCoaSupportEnabled'] = self.new_object.get('radiusCoaSupportEnabled')
        if self.new_object.get('radiusAccountingEnabled') is not None or self.new_object.get('radius_accounting_enabled') is not None:
            new_object_params['radiusAccountingEnabled'] = self.new_object.get('radiusAccountingEnabled')
        if self.new_object.get('radiusAccountingServers') is not None or self.new_object.get('radius_accounting_servers') is not None:
            new_object_params['radiusAccountingServers'] = self.new_object.get('radiusAccountingServers') or self.new_object.get('radius_accounting_servers')
        if self.new_object.get('radiusGroupAttribute') is not None or self.new_object.get('radius_group_attribute') is not None:
            new_object_params['radiusGroupAttribute'] = self.new_object.get('radiusGroupAttribute') or self.new_object.get('radius_group_attribute')
        if self.new_object.get('hostMode') is not None or self.new_object.get('host_mode') is not None:
            new_object_params['hostMode'] = self.new_object.get('hostMode') or self.new_object.get('host_mode')
        if self.new_object.get('accessPolicyType') is not None or self.new_object.get('access_policy_type') is not None:
            new_object_params['accessPolicyType'] = self.new_object.get('accessPolicyType') or self.new_object.get('access_policy_type')
        if self.new_object.get('increaseAccessSpeed') is not None or self.new_object.get('increase_access_speed') is not None:
            new_object_params['increaseAccessSpeed'] = self.new_object.get('increaseAccessSpeed')
        if self.new_object.get('guestVlanId') is not None or self.new_object.get('guest_vlan_id') is not None:
            new_object_params['guestVlanId'] = self.new_object.get('guestVlanId') or self.new_object.get('guest_vlan_id')
        if self.new_object.get('dot1x') is not None or self.new_object.get('dot1x') is not None:
            new_object_params['dot1x'] = self.new_object.get('dot1x') or self.new_object.get('dot1x')
        if self.new_object.get('voiceVlanClients') is not None or self.new_object.get('voice_vlan_clients') is not None:
            new_object_params['voiceVlanClients'] = self.new_object.get('voiceVlanClients')
        if self.new_object.get('urlRedirectWalledGardenEnabled') is not None or self.new_object.get('url_redirect_walled_garden_enabled') is not None:
            new_object_params['urlRedirectWalledGardenEnabled'] = self.new_object.get('urlRedirectWalledGardenEnabled')
        if self.new_object.get('urlRedirectWalledGardenRanges') is not None or self.new_object.get('url_redirect_walled_garden_ranges') is not None:
            new_object_params['urlRedirectWalledGardenRanges'] = self.new_object.get('urlRedirectWalledGardenRanges') or self.new_object.get('url_redirect_walled_garden_ranges')
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        if self.new_object.get('accessPolicyNumber') is not None or self.new_object.get('access_policy_number') is not None:
            new_object_params['accessPolicyNumber'] = self.new_object.get('accessPolicyNumber') or self.new_object.get('access_policy_number')
        return new_object_params

    def get_object_by_name(self, name):
        result = None
        try:
            items = self.meraki.exec_meraki(family='switch', function='getNetworkSwitchAccessPolicies', params=self.get_all_params(name=name))
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
        try:
            items = self.meraki.exec_meraki(family='switch', function='getNetworkSwitchAccessPolicy', params=self.get_params_by_id())
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
            result = get_dict_result(items, 'accessPolicyNumber', id)
        except Exception as e:
            print('Error: ', e)
            result = None
        return result

    def exists(self):
        id_exists = False
        name_exists = False
        prev_obj = None
        o_id = self.new_object.get('id')
        o_id = o_id or self.new_object.get('access_policy_number') or self.new_object.get('accessPolicyNumber')
        name = self.new_object.get('name')
        if o_id:
            prev_obj = self.get_object_by_id(o_id)
            id_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if not id_exists and name:
            prev_obj = self.get_object_by_name(name)
            name_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if name_exists:
            _id = prev_obj.get('id')
            _id = _id or prev_obj.get('accessPolicyNumber')
            if id_exists and name_exists and (o_id != _id):
                raise InconsistentParameters("The 'id' and 'name' params don't refer to the same object")
            if _id:
                self.new_object.update(dict(id=_id))
                self.new_object.update(dict(accessPolicyNumber=_id))
            if _id:
                prev_obj = self.get_object_by_id(_id)
        it_exists = prev_obj is not None and isinstance(prev_obj, dict)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('name', 'name'), ('radiusServers', 'radiusServers'), ('radius', 'radius'), ('guestPortBouncing', 'guestPortBouncing'), ('radiusTestingEnabled', 'radiusTestingEnabled'), ('radiusCoaSupportEnabled', 'radiusCoaSupportEnabled'), ('radiusAccountingEnabled', 'radiusAccountingEnabled'), ('radiusAccountingServers', 'radiusAccountingServers'), ('radiusGroupAttribute', 'radiusGroupAttribute'), ('hostMode', 'hostMode'), ('accessPolicyType', 'accessPolicyType'), ('increaseAccessSpeed', 'increaseAccessSpeed'), ('guestVlanId', 'guestVlanId'), ('dot1x', 'dot1x'), ('voiceVlanClients', 'voiceVlanClients'), ('urlRedirectWalledGardenEnabled', 'urlRedirectWalledGardenEnabled'), ('urlRedirectWalledGardenRanges', 'urlRedirectWalledGardenRanges'), ('networkId', 'networkId'), ('accessPolicyNumber', 'accessPolicyNumber')]
        return any((not meraki_compare_equality(current_obj.get(meraki_param), requested_obj.get(ansible_param)) for meraki_param, ansible_param in obj_params))

    def create(self):
        result = self.meraki.exec_meraki(family='switch', function='createNetworkSwitchAccessPolicy', params=self.create_params(), op_modifies=True)
        return result

    def update(self):
        id = self.new_object.get('id')
        id = id or self.new_object.get('accessPolicyNumber')
        name = self.new_object.get('name')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
                id_ = id_ or prev_obj_name.get('accessPolicyNumber')
            if id_:
                self.new_object.update(dict(accessPolicyNumber=id_))
        result = self.meraki.exec_meraki(family='switch', function='updateNetworkSwitchAccessPolicy', params=self.update_by_id_params(), op_modifies=True)
        return result

    def delete(self):
        id = self.new_object.get('id')
        id = id or self.new_object.get('accessPolicyNumber')
        name = self.new_object.get('name')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
                id_ = id_ or prev_obj_name.get('accessPolicyNumber')
            if id_:
                self.new_object.update(dict(accessPolicyNumber=id_))
        result = self.meraki.exec_meraki(family='switch', function='deleteNetworkSwitchAccessPolicy', params=self.delete_by_id_params())
        return result