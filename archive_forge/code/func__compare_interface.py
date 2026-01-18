from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.facts.facts import Facts
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.rm_templates.ospf_interfaces import (
def _compare_interface(self, want, have):
    wdict = want.get('address_family', {})
    hdict = have.get('address_family', {})
    wname = want.get('name')
    hname = have.get('name')
    h_value = {}
    for key, w_value in iteritems(wdict):
        if hdict and hdict.get(key):
            h_value = hdict[key]
        else:
            h_value = None
        w = {'name': wname, 'type': w_value['afi'], 'address_family': w_value}
        if h_value is not None:
            h = {'name': hname, 'type': h_value['afi'], 'address_family': h_value}
        else:
            h = {}
        self.compare(parsers='name', want=w, have=h)