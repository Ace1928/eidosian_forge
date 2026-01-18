from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.bgp_global import (
def _compare_redistribute_lists(self, want, have):
    """Compare redistribute list of dict"""
    redist_parses = ['application', 'bgp', 'connected', 'eigrp', 'isis', 'iso_igrp', 'lisp', 'mobile', 'odr', 'ospf', 'ospfv3', 'rip', 'static', 'vrf']
    for name, w_redist in want.items():
        have_nbr = have.pop(name, {})
        self.compare(parsers=redist_parses, want=w_redist, have=have_nbr)
    for name, h_redist in have.items():
        self.compare(parsers=redist_parses, want={}, have=h_redist)