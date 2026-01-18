from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.dnac.plugins.plugin_utils.dnac import (
from ansible_collections.cisco.dnac.plugins.plugin_utils.exceptions import (
class SdaVirtualNetworkIpPool(object):

    def __init__(self, params, dnac):
        self.dnac = dnac
        self.new_object = dict(site_name_hierarchy=params.get('siteNameHierarchy'), siteNameHierarchy=params.get('siteNameHierarchy'), virtualNetworkName=params.get('virtualNetworkName'), isLayer2Only=params.get('isLayer2Only'), ipPoolName=params.get('ipPoolName'), vlanId=params.get('vlanId'), vlanName=params.get('vlanName'), autoGenerateVlanName=params.get('autoGenerateVlanName'), trafficType=params.get('trafficType'), scalableGroupName=params.get('scalableGroupName'), isL2FloodingEnabled=params.get('isL2FloodingEnabled'), isThisCriticalPool=params.get('isThisCriticalPool'), isWirelessPool=params.get('isWirelessPool'), isIpDirectedBroadcast=params.get('isIpDirectedBroadcast'), isCommonPool=params.get('isCommonPool'), isBridgeModeVm=params.get('isBridgeModeVm'), poolType=params.get('poolType'), virtual_network_name=params.get('virtualNetworkName'), ip_pool_name=params.get('ipPoolName'))

    def get_all_params(self, name=None, id=None):
        new_object_params = {}
        new_object_params['site_name_hierarchy'] = self.new_object.get('site_name_hierarchy') or self.new_object.get('siteNameHierarchy')
        new_object_params['virtual_network_name'] = self.new_object.get('virtualNetworkName') or self.new_object.get('virtual_network_name')
        new_object_params['ip_pool_name'] = self.new_object.get('ipPoolName') or self.new_object.get('ip_pool_name')
        return new_object_params

    def create_params(self):
        new_object_params = {}
        new_object_params['siteNameHierarchy'] = self.new_object.get('siteNameHierarchy')
        new_object_params['virtualNetworkName'] = self.new_object.get('virtualNetworkName')
        new_object_params['isLayer2Only'] = self.new_object.get('isLayer2Only')
        new_object_params['ipPoolName'] = self.new_object.get('ipPoolName')
        new_object_params['vlanId'] = self.new_object.get('vlanId')
        new_object_params['vlanName'] = self.new_object.get('vlanName')
        new_object_params['autoGenerateVlanName'] = self.new_object.get('autoGenerateVlanName')
        new_object_params['trafficType'] = self.new_object.get('trafficType')
        new_object_params['scalableGroupName'] = self.new_object.get('scalableGroupName')
        new_object_params['isL2FloodingEnabled'] = self.new_object.get('isL2FloodingEnabled')
        new_object_params['isThisCriticalPool'] = self.new_object.get('isThisCriticalPool')
        new_object_params['isWirelessPool'] = self.new_object.get('isWirelessPool')
        new_object_params['isIpDirectedBroadcast'] = self.new_object.get('isIpDirectedBroadcast')
        new_object_params['isCommonPool'] = self.new_object.get('isCommonPool')
        new_object_params['isBridgeModeVm'] = self.new_object.get('isBridgeModeVm')
        new_object_params['poolType'] = self.new_object.get('poolType')
        return new_object_params

    def delete_all_params(self):
        new_object_params = {}
        new_object_params['siteNameHierarchy'] = self.new_object.get('site_name_hierarchy')
        new_object_params['site_name_hierarchy'] = self.new_object.get('site_name_hierarchy')
        new_object_params['virtual_network_name'] = self.new_object.get('virtual_network_name')
        new_object_params['ip_pool_name'] = self.new_object.get('ip_pool_name')
        return new_object_params

    def get_object_by_name(self, name, is_absent=False):
        result = None
        try:
            items = self.dnac.exec(family='sda', function='get_ip_pool_from_sda_virtual_network', params=self.get_all_params(name=name))
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
        name = self.new_object.get('name')
        prev_obj = self.get_object_by_name(name, is_absent=is_absent)
        it_exists = prev_obj is not None and isinstance(prev_obj, dict) and (prev_obj.get('status') != 'failed')
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('siteNameHierarchy', 'siteNameHierarchy'), ('virtualNetworkName', 'virtualNetworkName'), ('isLayer2Only', 'isLayer2Only'), ('ipPoolName', 'ipPoolName'), ('vlanId', 'vlanId'), ('vlanName', 'vlanName'), ('autoGenerateVlanName', 'autoGenerateVlanName'), ('trafficType', 'trafficType'), ('scalableGroupName', 'scalableGroupName'), ('isL2FloodingEnabled', 'isL2FloodingEnabled'), ('isThisCriticalPool', 'isThisCriticalPool'), ('isWirelessPool', 'isWirelessPool'), ('isIpDirectedBroadcast', 'isIpDirectedBroadcast'), ('isCommonPool', 'isCommonPool'), ('isBridgeModeVm', 'isBridgeModeVm'), ('poolType', 'poolType'), ('siteNameHierarchy', 'site_name_hierarchy'), ('virtualNetworkName', 'virtual_network_name'), ('ipPoolName', 'ip_pool_name')]
        return any((not dnac_compare_equality(current_obj.get(dnac_param), requested_obj.get(ansible_param)) for dnac_param, ansible_param in obj_params))

    def create(self):
        result = self.dnac.exec(family='sda', function='add_ip_pool_in_sda_virtual_network', params=self.create_params(), op_modifies=True)
        if isinstance(result, dict):
            if 'response' in result:
                result = result.get('response')
            if isinstance(result, dict) and result.get('status') == 'failed':
                raise AnsibleSDAException(response=result)
        return result

    def delete(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        result = self.dnac.exec(family='sda', function='delete_ip_pool_from_sda_virtual_network', params=self.delete_all_params())
        return result