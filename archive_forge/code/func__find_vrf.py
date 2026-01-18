from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.facts.facts import Facts
def _find_vrf(self, item, entries):
    """This method iterates through the items
            in `entries` and returns the object that
            matches `item`.

        :rtype: A dict
        :returns: the obj in `entries` that matches `item`
        """
    obj = {}
    afi = item.get('vrf')
    if afi:
        obj = search_obj_in_list(afi, entries, key='vrf') or {}
    else:
        for x in entries:
            if 'vrf' not in remove_empties(x):
                obj = x
                break
    return obj