from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.dnac.plugins.plugin_utils.dnac import (
from ansible_collections.cisco.dnac.plugins.plugin_utils.exceptions import (
class TransitPeerNetwork(object):

    def __init__(self, params, dnac):
        self.dnac = dnac
        self.new_object = dict(transitPeerNetworkName=params.get('transitPeerNetworkName'), transitPeerNetworkType=params.get('transitPeerNetworkType'), ipTransitSettings=params.get('ipTransitSettings'), sdaTransitSettings=params.get('sdaTransitSettings'), transit_peer_network_name=params.get('transitPeerNetworkName'))

    def get_all_params(self, name=None, id=None):
        new_object_params = {}
        new_object_params['transit_peer_network_name'] = self.new_object.get('transitPeerNetworkName') or self.new_object.get('transit_peer_network_name')
        return new_object_params

    def create_params(self):
        new_object_params = {}
        new_object_params['transitPeerNetworkName'] = self.new_object.get('transitPeerNetworkName')
        new_object_params['transitPeerNetworkType'] = self.new_object.get('transitPeerNetworkType')
        new_object_params['ipTransitSettings'] = self.new_object.get('ipTransitSettings')
        new_object_params['sdaTransitSettings'] = self.new_object.get('sdaTransitSettings')
        return new_object_params

    def delete_all_params(self):
        new_object_params = {}
        new_object_params['transit_peer_network_name'] = self.new_object.get('transit_peer_network_name')
        return new_object_params

    def get_object_by_name(self, name, is_absent=False):
        result = None
        try:
            items = self.dnac.exec(family='sda', function='get_transit_peer_network_info', params=self.get_all_params(name=name))
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
                if isinstance(items, dict) and items.get('status') == 'failed':
                    if is_absent:
                        raise AnsibleSDAException(response=items)
                    result = None
                    return result
            result = get_dict_result(items, 'name', name)
        except Exception:
            if is_absent:
                raise
            result = None
        return result

    def get_object_by_id(self, id):
        result = None
        return result

    def exists(self, is_absent=False):
        name = self.new_object.get('transitPeerNetworkName')
        prev_obj = self.get_object_by_name(name, is_absent=is_absent)
        it_exists = prev_obj is not None and isinstance(prev_obj, dict) and (prev_obj.get('status') != 'failed')
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('transitPeerNetworkName', 'transitPeerNetworkName'), ('transitPeerNetworkType', 'transitPeerNetworkType'), ('ipTransitSettings', 'ipTransitSettings'), ('sdaTransitSettings', 'sdaTransitSettings'), ('transitPeerNetworkName', 'transit_peer_network_name')]
        return any((not dnac_compare_equality(current_obj.get(dnac_param), requested_obj.get(ansible_param)) for dnac_param, ansible_param in obj_params))

    def create(self):
        result = self.dnac.exec(family='sda', function='add_transit_peer_network', params=self.create_params(), op_modifies=True)
        return result

    def delete(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        result = self.dnac.exec(family='sda', function='delete_transit_peer_network', params=self.delete_all_params())
        return result