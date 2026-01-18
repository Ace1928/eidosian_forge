from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.service import (
def _compare_lists_attrs(self, want, have):
    """Compare list of dict"""
    i_want = want.get('timestamps', {})
    i_have = have.get('timestamps', {})
    for key, wanting in iteritems(i_want):
        haveing = i_have.pop(key, {})
        if wanting != haveing:
            self.addcmd(wanting, 'timestamps')
    for key, haveing in iteritems(i_have):
        self.addcmd(haveing, 'timestamps', negate=True)