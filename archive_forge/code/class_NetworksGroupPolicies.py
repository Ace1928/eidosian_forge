from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.meraki.plugins.plugin_utils.meraki import (
from ansible_collections.cisco.meraki.plugins.plugin_utils.exceptions import (
class NetworksGroupPolicies(object):

    def __init__(self, params, meraki):
        self.meraki = meraki
        self.new_object = dict(name=params.get('name'), scheduling=params.get('scheduling'), bandwidth=params.get('bandwidth'), firewallAndTrafficShaping=params.get('firewallAndTrafficShaping'), contentFiltering=params.get('contentFiltering'), splashAuthSettings=params.get('splashAuthSettings'), vlanTagging=params.get('vlanTagging'), bonjourForwarding=params.get('bonjourForwarding'), networkId=params.get('networkId'), groupPolicyId=params.get('groupPolicyId'))

    def get_all_params(self, name=None, id=None):
        new_object_params = {}
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        return new_object_params

    def get_params_by_id(self, name=None, id=None):
        new_object_params = {}
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        if self.new_object.get('groupPolicyId') is not None or self.new_object.get('group_policy_id') is not None:
            new_object_params['groupPolicyId'] = self.new_object.get('groupPolicyId') or self.new_object.get('group_policy_id')
        return new_object_params

    def create_params(self):
        new_object_params = {}
        if self.new_object.get('name') is not None or self.new_object.get('name') is not None:
            new_object_params['name'] = self.new_object.get('name') or self.new_object.get('name')
        if self.new_object.get('scheduling') is not None or self.new_object.get('scheduling') is not None:
            new_object_params['scheduling'] = self.new_object.get('scheduling') or self.new_object.get('scheduling')
        if self.new_object.get('bandwidth') is not None or self.new_object.get('bandwidth') is not None:
            new_object_params['bandwidth'] = self.new_object.get('bandwidth') or self.new_object.get('bandwidth')
        if self.new_object.get('firewallAndTrafficShaping') is not None or self.new_object.get('firewall_and_traffic_shaping') is not None:
            new_object_params['firewallAndTrafficShaping'] = self.new_object.get('firewallAndTrafficShaping') or self.new_object.get('firewall_and_traffic_shaping')
        if self.new_object.get('contentFiltering') is not None or self.new_object.get('content_filtering') is not None:
            new_object_params['contentFiltering'] = self.new_object.get('contentFiltering') or self.new_object.get('content_filtering')
        if self.new_object.get('splashAuthSettings') is not None or self.new_object.get('splash_auth_settings') is not None:
            new_object_params['splashAuthSettings'] = self.new_object.get('splashAuthSettings') or self.new_object.get('splash_auth_settings')
        if self.new_object.get('vlanTagging') is not None or self.new_object.get('vlan_tagging') is not None:
            new_object_params['vlanTagging'] = self.new_object.get('vlanTagging') or self.new_object.get('vlan_tagging')
        if self.new_object.get('bonjourForwarding') is not None or self.new_object.get('bonjour_forwarding') is not None:
            new_object_params['bonjourForwarding'] = self.new_object.get('bonjourForwarding') or self.new_object.get('bonjour_forwarding')
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        return new_object_params

    def delete_by_id_params(self):
        new_object_params = {}
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        if self.new_object.get('groupPolicyId') is not None or self.new_object.get('group_policy_id') is not None:
            new_object_params['groupPolicyId'] = self.new_object.get('groupPolicyId') or self.new_object.get('group_policy_id')
        return new_object_params

    def update_by_id_params(self):
        new_object_params = {}
        if self.new_object.get('name') is not None or self.new_object.get('name') is not None:
            new_object_params['name'] = self.new_object.get('name') or self.new_object.get('name')
        if self.new_object.get('scheduling') is not None or self.new_object.get('scheduling') is not None:
            new_object_params['scheduling'] = self.new_object.get('scheduling') or self.new_object.get('scheduling')
        if self.new_object.get('bandwidth') is not None or self.new_object.get('bandwidth') is not None:
            new_object_params['bandwidth'] = self.new_object.get('bandwidth') or self.new_object.get('bandwidth')
        if self.new_object.get('firewallAndTrafficShaping') is not None or self.new_object.get('firewall_and_traffic_shaping') is not None:
            new_object_params['firewallAndTrafficShaping'] = self.new_object.get('firewallAndTrafficShaping') or self.new_object.get('firewall_and_traffic_shaping')
        if self.new_object.get('contentFiltering') is not None or self.new_object.get('content_filtering') is not None:
            new_object_params['contentFiltering'] = self.new_object.get('contentFiltering') or self.new_object.get('content_filtering')
        if self.new_object.get('splashAuthSettings') is not None or self.new_object.get('splash_auth_settings') is not None:
            new_object_params['splashAuthSettings'] = self.new_object.get('splashAuthSettings') or self.new_object.get('splash_auth_settings')
        if self.new_object.get('vlanTagging') is not None or self.new_object.get('vlan_tagging') is not None:
            new_object_params['vlanTagging'] = self.new_object.get('vlanTagging') or self.new_object.get('vlan_tagging')
        if self.new_object.get('bonjourForwarding') is not None or self.new_object.get('bonjour_forwarding') is not None:
            new_object_params['bonjourForwarding'] = self.new_object.get('bonjourForwarding') or self.new_object.get('bonjour_forwarding')
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        if self.new_object.get('groupPolicyId') is not None or self.new_object.get('group_policy_id') is not None:
            new_object_params['groupPolicyId'] = self.new_object.get('groupPolicyId') or self.new_object.get('group_policy_id')
        return new_object_params

    def get_object_by_name(self, name):
        result = None
        try:
            items = self.meraki.exec_meraki(family='networks', function='getNetworkGroupPolicies', params=self.get_all_params(name=name))
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
            items = self.meraki.exec_meraki(family='networks', function='getNetworkGroupPolicy', params=self.get_params_by_id())
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
            result = get_dict_result(items, 'groupPolicyId', id)
        except Exception as e:
            print('Error: ', e)
            result = None
        return result

    def exists(self):
        id_exists = False
        name_exists = False
        prev_obj = None
        o_id = self.new_object.get('id')
        o_id = o_id or self.new_object.get('group_policy_id') or self.new_object.get('groupPolicyId')
        name = self.new_object.get('name')
        if o_id:
            prev_obj = self.get_object_by_id(o_id)
            id_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if not id_exists and name:
            prev_obj = self.get_object_by_name(name)
            name_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if name_exists:
            _id = prev_obj.get('id')
            _id = _id or prev_obj.get('groupPolicyId')
            if id_exists and name_exists and (o_id != _id):
                raise InconsistentParameters("The 'id' and 'name' params don't refer to the same object")
            if _id:
                self.new_object.update(dict(id=_id))
                self.new_object.update(dict(groupPolicyId=_id))
            if _id:
                prev_obj = self.get_object_by_id(_id)
        it_exists = prev_obj is not None and isinstance(prev_obj, dict)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('name', 'name'), ('scheduling', 'scheduling'), ('bandwidth', 'bandwidth'), ('firewallAndTrafficShaping', 'firewallAndTrafficShaping'), ('contentFiltering', 'contentFiltering'), ('splashAuthSettings', 'splashAuthSettings'), ('vlanTagging', 'vlanTagging'), ('bonjourForwarding', 'bonjourForwarding'), ('networkId', 'networkId'), ('groupPolicyId', 'groupPolicyId')]
        return any((not meraki_compare_equality(current_obj.get(meraki_param), requested_obj.get(ansible_param)) for meraki_param, ansible_param in obj_params))

    def create(self):
        result = self.meraki.exec_meraki(family='networks', function='createNetworkGroupPolicy', params=self.create_params(), op_modifies=True)
        return result

    def update(self):
        id = self.new_object.get('id')
        id = id or self.new_object.get('groupPolicyId')
        name = self.new_object.get('name')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
                id_ = id_ or prev_obj_name.get('groupPolicyId')
            if id_:
                self.new_object.update(dict(groupPolicyId=id_))
        result = self.meraki.exec_meraki(family='networks', function='updateNetworkGroupPolicy', params=self.update_by_id_params(), op_modifies=True)
        return result

    def delete(self):
        id = self.new_object.get('id')
        id = id or self.new_object.get('groupPolicyId')
        name = self.new_object.get('name')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
                id_ = id_ or prev_obj_name.get('groupPolicyId')
            if id_:
                self.new_object.update(dict(groupPolicyId=id_))
        result = self.meraki.exec_meraki(family='networks', function='deleteNetworkGroupPolicy', params=self.delete_by_id_params())
        return result