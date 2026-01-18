from __future__ import absolute_import, division, print_function
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.ise.plugins.plugin_utils.ise import (
from ansible_collections.cisco.ise.plugins.plugin_utils.exceptions import (
class AciSettings(object):

    def __init__(self, params, ise):
        self.ise = ise
        self.new_object = dict(id=params.get('id'), enable_aci=params.get('enableAci'), ip_address_host_name=params.get('ipAddressHostName'), admin_name=params.get('adminName'), admin_password=params.get('adminPassword'), aciipaddress=params.get('aciipaddress'), aciuser_name=params.get('aciuserName'), acipassword=params.get('acipassword'), tenant_name=params.get('tenantName'), l3_route_network=params.get('l3RouteNetwork'), suffix_to_epg=params.get('suffixToEpg'), suffix_to_sgt=params.get('suffixToSgt'), all_sxp_domain=params.get('allSxpDomain'), specific_sxp_domain=params.get('specificSxpDomain'), specifix_sxp_domain_list=params.get('specifixSxpDomainList'), enable_data_plane=params.get('enableDataPlane'), untagged_packet_iepg_name=params.get('untaggedPacketIepgName'), default_sgt_name=params.get('defaultSgtName'), enable_elements_limit=params.get('enableElementsLimit'), max_num_iepg_from_aci=params.get('maxNumIepgFromAci'), max_num_sgt_to_aci=params.get('maxNumSgtToAci'), aci50=params.get('aci50'), aci51=params.get('aci51'))

    def get_object_by_name(self, name):
        result = None
        items = self.ise.exec(family='aci_settings', function='get_aci_settings').response['AciSettings']
        result = get_dict_result(items, 'name', name)
        return result

    def get_object_by_id(self, id):
        try:
            result = self.ise.exec(family='aci_settings', function='get_aci_settings', handle_func_exception=False).response['AciSettings']
        except Exception as e:
            result = None
        return result

    def exists(self):
        prev_obj = None
        id_exists = False
        name_exists = False
        o_id = self.new_object.get('id')
        name = self.new_object.get('name')
        if o_id:
            prev_obj = self.get_object_by_id(o_id)
            id_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if not id_exists and name:
            prev_obj = self.get_object_by_name(name)
            name_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if name_exists:
            _id = prev_obj.get('id')
            if id_exists and name_exists and (o_id != _id):
                raise InconsistentParameters("The 'id' and 'name' params don't refer to the same object")
        it_exists = prev_obj is not None and isinstance(prev_obj, dict)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('id', 'id'), ('enableAci', 'enable_aci'), ('ipAddressHostName', 'ip_address_host_name'), ('adminName', 'admin_name'), ('adminPassword', 'admin_password'), ('aciipaddress', 'aciipaddress'), ('aciuserName', 'aciuser_name'), ('acipassword', 'acipassword'), ('tenantName', 'tenant_name'), ('l3RouteNetwork', 'l3_route_network'), ('suffixToEpg', 'suffix_to_epg'), ('suffixToSgt', 'suffix_to_sgt'), ('allSxpDomain', 'all_sxp_domain'), ('specificSxpDomain', 'specific_sxp_domain'), ('specifixSxpDomainList', 'specifix_sxp_domain_list'), ('enableDataPlane', 'enable_data_plane'), ('untaggedPacketIepgName', 'untagged_packet_iepg_name'), ('defaultSgtName', 'default_sgt_name'), ('enableElementsLimit', 'enable_elements_limit'), ('maxNumIepgFromAci', 'max_num_iepg_from_aci'), ('maxNumSgtToAci', 'max_num_sgt_to_aci'), ('aci50', 'aci50'), ('aci51', 'aci51')]
        return any((not ise_compare_equality(current_obj.get(ise_param), requested_obj.get(ansible_param)) for ise_param, ansible_param in obj_params))

    def update(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        if not id:
            id_ = self.get_object_by_name(name).get('id')
            self.new_object.update(dict(id=id_))
        result = self.ise.exec(family='aci_settings', function='update_aci_settings_by_id', params=self.new_object).response
        return result