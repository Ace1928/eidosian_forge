from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.facts.facts import Facts
def _static_route_popper(self, want_afi, have_afi):
    """ """
    commands = []
    update_commands = []
    if not want_afi.get('routes', []):
        commands.append('no address-family {0} {1}'.format(have_afi['afi'], have_afi['safi']))
    else:
        for have_route in have_afi.get('routes', []):
            want_route = search_obj_in_list(have_route['dest'], want_afi.get('routes', []), key='dest') or {}
            rotated_want_next_hops = self.rotate_next_hops(want_route.get('next_hops', {}))
            rotated_have_next_hops = self.rotate_next_hops(have_route.get('next_hops', {}))
            for key in rotated_want_next_hops.keys():
                if key in rotated_have_next_hops:
                    cmd = 'no {0}'.format(want_route['dest'])
                    for item in key:
                        if '.' in item or ':' in item or '/' in item:
                            cmd += ' {0}'.format(item)
                        else:
                            cmd += ' vrf {0}'.format(item)
                    update_commands.append(cmd)
        if update_commands:
            update_commands.insert(0, 'address-family {0} {1}'.format(have_afi['afi'], have_afi['safi']))
            commands.extend(update_commands)
    return commands