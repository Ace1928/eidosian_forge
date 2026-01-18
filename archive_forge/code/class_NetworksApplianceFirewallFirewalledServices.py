from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.meraki.plugins.plugin_utils.meraki import (
from ansible_collections.cisco.meraki.plugins.plugin_utils.exceptions import (
class NetworksApplianceFirewallFirewalledServices(object):

    def __init__(self, params, meraki):
        self.meraki = meraki
        self.new_object = dict(access=params.get('access'), allowedIps=params.get('allowedIps'), network_id=params.get('networkId'), service=params.get('service'))

    def get_params_by_id(self, name=None, id=None):
        new_object_params = {}
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        if self.new_object.get('service') is not None or self.new_object.get('service') is not None:
            new_object_params['service'] = self.new_object.get('service')
        return new_object_params

    def update_by_id_params(self):
        new_object_params = {}
        if self.new_object.get('access') is not None or self.new_object.get('access') is not None:
            new_object_params['access'] = self.new_object.get('access') or self.new_object.get('access')
        if self.new_object.get('allowedIps') is not None or self.new_object.get('allowed_ips') is not None:
            new_object_params['allowedIps'] = self.new_object.get('allowedIps') or self.new_object.get('allowed_ips')
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        if self.new_object.get('service') is not None or self.new_object.get('service') is not None:
            new_object_params['service'] = self.new_object.get('service') or self.new_object.get('service')
        return new_object_params

    def get_object_by_name(self, name):
        result = None
        return result

    def get_object_by_id(self, id):
        result = None
        try:
            items = self.meraki.exec_meraki(family='appliance', function='getNetworkApplianceFirewallFirewalledService', params=self.get_params_by_id())
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
        o_id = o_id or self.new_object.get('service') or self.new_object.get('service')
        name = self.new_object.get('name')
        if o_id:
            prev_obj = self.get_object_by_id(o_id)
            id_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if not id_exists and name:
            prev_obj = self.get_object_by_name(name)
            name_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if name_exists:
            _id = prev_obj.get('id')
            _id = _id or prev_obj.get('service')
            if id_exists and name_exists and (o_id != _id):
                raise InconsistentParameters("The 'id' and 'name' params don't refer to the same object")
            if _id:
                self.new_object.update(dict(id=_id))
                self.new_object.update(dict(service=_id))
            if _id:
                prev_obj = self.get_object_by_id(_id)
        it_exists = prev_obj is not None and isinstance(prev_obj, dict)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('access', 'access'), ('allowedIps', 'allowedIps'), ('networkId', 'networkId'), ('service', 'service')]
        return any((not meraki_compare_equality(current_obj.get(meraki_param), requested_obj.get(ansible_param)) for meraki_param, ansible_param in obj_params))

    def update(self):
        id = self.new_object.get('id')
        id = id or self.new_object.get('service')
        name = self.new_object.get('name')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
                id_ = id_ or prev_obj_name.get('service')
            if id_:
                self.new_object.update(dict(service=id_))
        result = self.meraki.exec_meraki(family='appliance', function='updateNetworkApplianceFirewallFirewalledService', params=self.update_by_id_params(), op_modifies=True)
        return result