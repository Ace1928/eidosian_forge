from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.meraki.plugins.plugin_utils.meraki import (
from ansible_collections.cisco.meraki.plugins.plugin_utils.exceptions import (
class NetworksSwitchQosRulesOrder(object):

    def __init__(self, params, meraki):
        self.meraki = meraki
        self.new_object = dict(vlan=params.get('vlan'), protocol=params.get('protocol'), srcPort=params.get('srcPort'), srcPortRange=params.get('srcPortRange'), dstPort=params.get('dstPort'), dstPortRange=params.get('dstPortRange'), dscp=params.get('dscp'), networkId=params.get('networkId'), qosRuleId=params.get('qosRuleId'))

    def get_all_params(self, name=None, id=None):
        new_object_params = {}
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        return new_object_params

    def get_params_by_id(self, name=None, id=None):
        new_object_params = {}
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        if self.new_object.get('qosRuleId') is not None or self.new_object.get('qos_rule_id') is not None:
            new_object_params['qosRuleId'] = self.new_object.get('qosRuleId') or self.new_object.get('qos_rule_id')
        return new_object_params

    def create_params(self):
        new_object_params = {}
        if self.new_object.get('vlan') is not None or self.new_object.get('vlan') is not None:
            new_object_params['vlan'] = self.new_object.get('vlan') or self.new_object.get('vlan')
        if self.new_object.get('protocol') is not None or self.new_object.get('protocol') is not None:
            new_object_params['protocol'] = self.new_object.get('protocol') or self.new_object.get('protocol')
        if self.new_object.get('srcPort') is not None or self.new_object.get('src_port') is not None:
            new_object_params['srcPort'] = self.new_object.get('srcPort') or self.new_object.get('src_port')
        if self.new_object.get('srcPortRange') is not None or self.new_object.get('src_port_range') is not None:
            new_object_params['srcPortRange'] = self.new_object.get('srcPortRange') or self.new_object.get('src_port_range')
        if self.new_object.get('dstPort') is not None or self.new_object.get('dst_port') is not None:
            new_object_params['dstPort'] = self.new_object.get('dstPort') or self.new_object.get('dst_port')
        if self.new_object.get('dstPortRange') is not None or self.new_object.get('dst_port_range') is not None:
            new_object_params['dstPortRange'] = self.new_object.get('dstPortRange') or self.new_object.get('dst_port_range')
        if self.new_object.get('dscp') is not None or self.new_object.get('dscp') is not None:
            new_object_params['dscp'] = self.new_object.get('dscp') or self.new_object.get('dscp')
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        return new_object_params

    def delete_by_id_params(self):
        new_object_params = {}
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        if self.new_object.get('qosRuleId') is not None or self.new_object.get('qos_rule_id') is not None:
            new_object_params['qosRuleId'] = self.new_object.get('qosRuleId') or self.new_object.get('qos_rule_id')
        return new_object_params

    def update_by_id_params(self):
        new_object_params = {}
        if self.new_object.get('vlan') is not None or self.new_object.get('vlan') is not None:
            new_object_params['vlan'] = self.new_object.get('vlan') or self.new_object.get('vlan')
        if self.new_object.get('protocol') is not None or self.new_object.get('protocol') is not None:
            new_object_params['protocol'] = self.new_object.get('protocol') or self.new_object.get('protocol')
        if self.new_object.get('srcPort') is not None or self.new_object.get('src_port') is not None:
            new_object_params['srcPort'] = self.new_object.get('srcPort') or self.new_object.get('src_port')
        if self.new_object.get('srcPortRange') is not None or self.new_object.get('src_port_range') is not None:
            new_object_params['srcPortRange'] = self.new_object.get('srcPortRange') or self.new_object.get('src_port_range')
        if self.new_object.get('dstPort') is not None or self.new_object.get('dst_port') is not None:
            new_object_params['dstPort'] = self.new_object.get('dstPort') or self.new_object.get('dst_port')
        if self.new_object.get('dstPortRange') is not None or self.new_object.get('dst_port_range') is not None:
            new_object_params['dstPortRange'] = self.new_object.get('dstPortRange') or self.new_object.get('dst_port_range')
        if self.new_object.get('dscp') is not None or self.new_object.get('dscp') is not None:
            new_object_params['dscp'] = self.new_object.get('dscp') or self.new_object.get('dscp')
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        if self.new_object.get('qosRuleId') is not None or self.new_object.get('qos_rule_id') is not None:
            new_object_params['qosRuleId'] = self.new_object.get('qosRuleId') or self.new_object.get('qos_rule_id')
        return new_object_params

    def get_object_by_name(self, name):
        result = None
        try:
            items = self.meraki.exec_meraki(family='switch', function='getNetworkSwitchQosRules', params=self.get_all_params(name=name))
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
            items = self.meraki.exec_meraki(family='switch', function='getNetworkSwitchQosRule', params=self.get_params_by_id())
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
            result = get_dict_result(items, 'qosRuleId', id)
        except Exception as e:
            print('Error: ', e)
            result = None
        return result

    def exists(self):
        id_exists = False
        name_exists = False
        prev_obj = None
        o_id = self.new_object.get('id')
        o_id = o_id or self.new_object.get('qos_rule_id') or self.new_object.get('qosRuleId')
        name = self.new_object.get('name')
        if o_id:
            prev_obj = self.get_object_by_id(o_id)
            id_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if not id_exists and name:
            prev_obj = self.get_object_by_name(name)
            name_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if name_exists:
            _id = prev_obj.get('id')
            _id = _id or prev_obj.get('qosRuleId')
            if id_exists and name_exists and (o_id != _id):
                raise InconsistentParameters("The 'id' and 'name' params don't refer to the same object")
            if _id:
                self.new_object.update(dict(id=_id))
                self.new_object.update(dict(qosRuleId=_id))
            if _id:
                prev_obj = self.get_object_by_id(_id)
        it_exists = prev_obj is not None and isinstance(prev_obj, dict)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('vlan', 'vlan'), ('protocol', 'protocol'), ('srcPort', 'srcPort'), ('srcPortRange', 'srcPortRange'), ('dstPort', 'dstPort'), ('dstPortRange', 'dstPortRange'), ('dscp', 'dscp'), ('networkId', 'networkId'), ('qosRuleId', 'qosRuleId')]
        return any((not meraki_compare_equality(current_obj.get(meraki_param), requested_obj.get(ansible_param)) for meraki_param, ansible_param in obj_params))

    def create(self):
        result = self.meraki.exec_meraki(family='switch', function='createNetworkSwitchQosRule', params=self.create_params(), op_modifies=True)
        return result

    def update(self):
        id = self.new_object.get('id')
        id = id or self.new_object.get('qosRuleId')
        name = self.new_object.get('name')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
                id_ = id_ or prev_obj_name.get('qosRuleId')
            if id_:
                self.new_object.update(dict(qosRuleId=id_))
        result = self.meraki.exec_meraki(family='switch', function='updateNetworkSwitchQosRule', params=self.update_by_id_params(), op_modifies=True)
        return result

    def delete(self):
        id = self.new_object.get('id')
        id = id or self.new_object.get('qosRuleId')
        name = self.new_object.get('name')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
                id_ = id_ or prev_obj_name.get('qosRuleId')
            if id_:
                self.new_object.update(dict(qosRuleId=id_))
        result = self.meraki.exec_meraki(family='switch', function='deleteNetworkSwitchQosRule', params=self.delete_by_id_params())
        return result