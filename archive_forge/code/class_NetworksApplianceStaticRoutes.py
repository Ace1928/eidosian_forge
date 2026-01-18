from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.meraki.plugins.plugin_utils.meraki import (
from ansible_collections.cisco.meraki.plugins.plugin_utils.exceptions import (
class NetworksApplianceStaticRoutes(object):

    def __init__(self, params, meraki):
        self.meraki = meraki
        self.new_object = dict(name=params.get('name'), subnet=params.get('subnet'), gatewayIp=params.get('gatewayIp'), gatewayVlanId=params.get('gatewayVlanId'), networkId=params.get('networkId'), staticRouteId=params.get('staticRouteId'), enabled=params.get('enabled'), fixedIpAssignments=params.get('fixedIpAssignments'), reservedIpRanges=params.get('reservedIpRanges'))

    def get_all_params(self, name=None, id=None):
        new_object_params = {}
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        return new_object_params

    def get_params_by_id(self, name=None, id=None):
        new_object_params = {}
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        if self.new_object.get('staticRouteId') is not None or self.new_object.get('static_route_id') is not None:
            new_object_params['staticRouteId'] = self.new_object.get('staticRouteId') or self.new_object.get('static_route_id')
        return new_object_params

    def create_params(self):
        new_object_params = {}
        if self.new_object.get('name') is not None or self.new_object.get('name') is not None:
            new_object_params['name'] = self.new_object.get('name') or self.new_object.get('name')
        if self.new_object.get('subnet') is not None or self.new_object.get('subnet') is not None:
            new_object_params['subnet'] = self.new_object.get('subnet') or self.new_object.get('subnet')
        if self.new_object.get('gatewayIp') is not None or self.new_object.get('gateway_ip') is not None:
            new_object_params['gatewayIp'] = self.new_object.get('gatewayIp') or self.new_object.get('gateway_ip')
        if self.new_object.get('gatewayVlanId') is not None or self.new_object.get('gateway_vlan_id') is not None:
            new_object_params['gatewayVlanId'] = self.new_object.get('gatewayVlanId') or self.new_object.get('gateway_vlan_id')
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        return new_object_params

    def delete_by_id_params(self):
        new_object_params = {}
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        if self.new_object.get('staticRouteId') is not None or self.new_object.get('static_route_id') is not None:
            new_object_params['staticRouteId'] = self.new_object.get('staticRouteId') or self.new_object.get('static_route_id')
        return new_object_params

    def update_by_id_params(self):
        new_object_params = {}
        if self.new_object.get('name') is not None or self.new_object.get('name') is not None:
            new_object_params['name'] = self.new_object.get('name') or self.new_object.get('name')
        if self.new_object.get('subnet') is not None or self.new_object.get('subnet') is not None:
            new_object_params['subnet'] = self.new_object.get('subnet') or self.new_object.get('subnet')
        if self.new_object.get('gatewayIp') is not None or self.new_object.get('gateway_ip') is not None:
            new_object_params['gatewayIp'] = self.new_object.get('gatewayIp') or self.new_object.get('gateway_ip')
        if self.new_object.get('gatewayVlanId') is not None or self.new_object.get('gateway_vlan_id') is not None:
            new_object_params['gatewayVlanId'] = self.new_object.get('gatewayVlanId') or self.new_object.get('gateway_vlan_id')
        if self.new_object.get('enabled') is not None or self.new_object.get('enabled') is not None:
            new_object_params['enabled'] = self.new_object.get('enabled')
        if self.new_object.get('fixedIpAssignments') is not None or self.new_object.get('fixed_ip_assignments') is not None:
            new_object_params['fixedIpAssignments'] = self.new_object.get('fixedIpAssignments') or self.new_object.get('fixed_ip_assignments')
        if self.new_object.get('reservedIpRanges') is not None or self.new_object.get('reserved_ip_ranges') is not None:
            new_object_params['reservedIpRanges'] = self.new_object.get('reservedIpRanges') or self.new_object.get('reserved_ip_ranges')
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        if self.new_object.get('staticRouteId') is not None or self.new_object.get('static_route_id') is not None:
            new_object_params['staticRouteId'] = self.new_object.get('staticRouteId') or self.new_object.get('static_route_id')
        return new_object_params

    def get_object_by_name(self, name):
        result = None
        try:
            items = self.meraki.exec_meraki(family='appliance', function='getNetworkApplianceStaticRoutes', params=self.get_all_params(name=name))
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
            items = self.meraki.exec_meraki(family='appliance', function='getNetworkApplianceStaticRoute', params=self.get_params_by_id())
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
            result = items
        except Exception:
            result = None
        return result

    def exists(self):
        id_exists = False
        name_exists = False
        prev_obj = None
        o_id = self.new_object.get('id')
        o_id = o_id or self.new_object.get('static_route_id') or self.new_object.get('staticRouteId')
        name = self.new_object.get('name')
        if o_id:
            prev_obj = self.get_object_by_id(o_id)
            id_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if not id_exists and name:
            prev_obj = self.get_object_by_name(name)
            name_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if name_exists:
            _id = prev_obj.get('id')
            _id = _id or prev_obj.get('staticRouteId')
            if id_exists and name_exists and (o_id != _id):
                raise InconsistentParameters("The 'id' and 'name' params don't refer to the same object")
            if _id:
                self.new_object.update(dict(id=_id))
                self.new_object.update(dict(staticRouteId=_id))
            if _id:
                prev_obj = self.get_object_by_id(_id)
        it_exists = prev_obj is not None and isinstance(prev_obj, dict)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('name', 'name'), ('subnet', 'subnet'), ('gatewayIp', 'gatewayIp'), ('gatewayVlanId', 'gatewayVlanId'), ('networkId', 'networkId'), ('staticRouteId', 'staticRouteId'), ('enabled', 'enabled'), ('fixedIpAssignments', 'fixedIpAssignments'), ('reservedIpRanges', 'reservedIpRanges')]
        return any((not meraki_compare_equality(current_obj.get(meraki_param), requested_obj.get(ansible_param)) for meraki_param, ansible_param in obj_params))

    def create(self):
        result = self.meraki.exec_meraki(family='appliance', function='createNetworkApplianceStaticRoute', params=self.create_params(), op_modifies=True)
        return result

    def update(self):
        id = self.new_object.get('id')
        id = id or self.new_object.get('staticRouteId')
        name = self.new_object.get('name')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
                id_ = id_ or prev_obj_name.get('staticRouteId')
            if id_:
                self.new_object.update(dict(staticRouteId=id_))
        result = self.meraki.exec_meraki(family='appliance', function='updateNetworkApplianceStaticRoute', params=self.update_by_id_params(), op_modifies=True)
        return result

    def delete(self):
        id = self.new_object.get('id')
        id = id or self.new_object.get('staticRouteId')
        name = self.new_object.get('name')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
                id_ = id_ or prev_obj_name.get('staticRouteId')
            if id_:
                self.new_object.update(dict(staticRouteId=id_))
        result = self.meraki.exec_meraki(family='appliance', function='deleteNetworkApplianceStaticRoute', params=self.delete_by_id_params())
        return result