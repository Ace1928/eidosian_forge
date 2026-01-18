from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.dnac.plugins.plugin_utils.dnac import (
from ansible_collections.cisco.dnac.plugins.plugin_utils.exceptions import (
class SdaMulticast(object):

    def __init__(self, params, dnac):
        self.dnac = dnac
        self.new_object = dict(siteNameHierarchy=params.get('siteNameHierarchy'), multicastMethod=params.get('multicastMethod'), multicastType=params.get('multicastType'), multicastVnInfo=params.get('multicastVnInfo'), site_name_hierarchy=params.get('siteNameHierarchy'))

    def get_all_params(self, name=None, id=None):
        new_object_params = {}
        new_object_params['site_name_hierarchy'] = self.new_object.get('siteNameHierarchy') or self.new_object.get('site_name_hierarchy')
        return new_object_params

    def create_params(self):
        new_object_params = {}
        new_object_params['siteNameHierarchy'] = self.new_object.get('siteNameHierarchy')
        new_object_params['multicastMethod'] = self.new_object.get('multicastMethod')
        new_object_params['multicastType'] = self.new_object.get('multicastType')
        new_object_params['multicastVnInfo'] = self.new_object.get('multicastVnInfo')
        return new_object_params

    def delete_all_params(self):
        new_object_params = {}
        new_object_params['site_name_hierarchy'] = self.new_object.get('site_name_hierarchy')
        return new_object_params

    def get_object_by_name(self, name, is_absent=False):
        result = None
        try:
            items = self.dnac.exec(family='sda', function='get_multicast_details_from_sda_fabric', params=self.get_all_params(name=name))
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
        it_exists = prev_obj is not None and isinstance(prev_obj, dict) and ('siteNameHierarchy' in prev_obj)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('siteNameHierarchy', 'siteNameHierarchy'), ('multicastMethod', 'multicastMethod'), ('multicastType', 'multicastType'), ('multicastVnInfo', 'multicastVnInfo'), ('siteNameHierarchy', 'site_name_hierarchy')]
        return any((not dnac_compare_equality(current_obj.get(dnac_param), requested_obj.get(ansible_param)) for dnac_param, ansible_param in obj_params))

    def create(self):
        result = self.dnac.exec(family='sda', function='add_multicast_in_sda_fabric', params=self.create_params(), op_modifies=True)
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
        result = self.dnac.exec(family='sda', function='delete_multicast_from_sda_fabric', params=self.delete_all_params())
        return result