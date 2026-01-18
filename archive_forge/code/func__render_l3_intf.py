from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils._text import to_bytes
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import iteritems, string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.argspec.l3_interfaces.l3_interfaces import (
def _render_l3_intf(self, unit, int_dict):
    """

        :param item:
        :param int_dict:
        :return:
        """
    interface = {}
    ipv4 = []
    ipv6 = []
    if 'family' in unit.keys():
        if 'inet' in unit['family'].keys():
            interface['name'] = int_dict['name']
            interface['unit'] = unit['name']
            inet = unit['family'].get('inet')
            if inet is not None and 'address' in inet.keys():
                if isinstance(inet['address'], dict):
                    for key, value in iteritems(inet['address']):
                        addr = {}
                        addr['address'] = value
                        ipv4.append(addr)
                else:
                    for ip in inet['address']:
                        addr = {}
                        addr['address'] = ip['name']
                        ipv4.append(addr)
        if 'inet' in unit['family'].keys():
            interface['name'] = int_dict['name']
            interface['unit'] = unit['name']
            inet = unit['family'].get('inet')
            if inet is not None and 'dhcp' in inet.keys():
                addr = {}
                addr['address'] = 'dhcp'
                ipv4.append(addr)
        if 'inet6' in unit['family'].keys():
            interface['name'] = int_dict['name']
            interface['unit'] = unit['name']
            inet6 = unit['family'].get('inet6')
            if inet6 is not None and 'address' in inet6.keys():
                if isinstance(inet6['address'], dict):
                    for key, value in iteritems(inet6['address']):
                        addr = {}
                        addr['address'] = value
                        ipv6.append(addr)
                else:
                    for ip in inet6['address']:
                        addr = {}
                        addr['address'] = ip['name']
                        ipv4.append(addr)
        interface['ipv4'] = ipv4
        interface['ipv6'] = ipv6
    return utils.remove_empties(interface)