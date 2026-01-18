from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.bgp_address_family import (
def _bgp_add_fam_list_to_dict(self, tmp_data):
    """Convert all list of dicts to dicts of dicts, also deals with deprecated attributes"""
    p_key = {'aggregate_address': 'address', 'aggregate_addresses': 'address', 'neighbor': 'neighbor_address', 'neighbors': 'neighbor_address', 'route_maps': 'name', 'prefix_lists': 'name', 'networks': 'address', 'network': 'address', 'ospf': 'process_id', 'ospfv3': 'process_id'}
    af_data = {}
    for af in tmp_data:
        _af = {}
        for k, tval in af.items():
            val = deepcopy(tval)
            if k == 'neighbor' or k == 'neighbors':
                tmp = {}
                for neib in val:
                    if neib.get('address'):
                        neib['neighbor_address'] = neib.pop('address')
                    if neib.get('tag'):
                        neib['neighbor_address'] = neib.pop('tag')
                    if neib.get('ipv6_address'):
                        neib['neighbor_address'] = neib.pop('ipv6_address')
                    if neib.get('ipv6_adddress'):
                        neib['neighbor_address'] = neib.pop('ipv6_adddress')
                    if neib.get('prefix_list'):
                        neib['prefix_lists'] = [neib.pop('prefix_list')]
                    if neib.get('prefix_lists'):
                        neib['prefix_lists'] = {str(i[p_key['prefix_lists']]): i for i in neib.get('prefix_lists')}
                    if neib.get('route_map'):
                        neib['route_maps'] = [neib.pop('route_map')]
                    if neib.get('route_maps'):
                        neib['route_maps'] = {str(i[p_key['route_maps']]): i for i in neib.get('route_maps')}
                    if neib.get('slow_peer'):
                        neib['slow_peer_options'] = neib.pop('slow_peer')[0]
                    if neib.get('remote_as'):
                        neib['remote_as'] = str(neib.get('remote_as'))
                    if neib.get('local_as') and neib.get('local_as', {}).get('number'):
                        neib['local_as']['number'] = str(neib['local_as']['number'])
                    tmp[neib[p_key[k]]] = neib
                _af['neighbors'] = tmp
            elif k == 'network' or k == 'networks':
                _af['networks'] = {str(i[p_key[k]]): i for i in tval}
            elif k == 'aggregate_address' or k == 'aggregate_addresses':
                _af['aggregate_addresses'] = {str(i[p_key[k]]): i for i in tval}
            elif k == 'bgp':
                _af['bgp'] = val
                if val.get('slow_peer'):
                    _af['bgp']['slow_peer_options'] = _af['bgp'].pop('slow_peer')[0]
            elif k == 'redistribute':
                _redist = {}
                for i in tval:
                    if any((x in ['ospf', 'ospfv3'] for x in i)):
                        for ospf_version in ['ospf', 'ospfv3']:
                            if i.get(ospf_version):
                                _i = i[ospf_version]
                                if _i.get('match'):
                                    for depr in ['external', 'nssa_external', 'type_1', 'type_2']:
                                        if depr in _i['match'].keys():
                                            val = _i['match'].pop(depr, False)
                                            if depr.startswith('type'):
                                                if 'nssa_externals' in _i['match'].keys():
                                                    _i['match']['nssa_externals'][depr] = val
                                                else:
                                                    _i['match']['nssa_externals'] = {depr: val}
                                            elif depr in ['external', 'nssa_external']:
                                                _i['match'][depr + 's'] = {'type_1': True, 'type_2': True}
                                if ospf_version not in _redist:
                                    _redist[ospf_version] = {}
                                _redist[ospf_version].update({str(_i[p_key[ospf_version]]): dict(_i.items())})
                                break
                    else:
                        _redist.update(i)
                _af['redistribute'] = _redist
            else:
                _af[k] = tval
        af_data[af.get('afi', '') + '_' + af.get('safi', '') + '_' + af.get('vrf', '')] = _af
    return af_data