from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.logging_global import (
def _compare_complex_attrs(self, want, have):
    """Compare dict of list"""
    for _par in self.complex_parsers:
        i_want = want.get(_par, {})
        i_have = have.get(_par, {})
        for key, wanting in iteritems(i_want):
            haveing = i_have.pop(key, {})
            if wanting != haveing:
                if haveing and self.state in ['overridden', 'replaced']:
                    self.addcmd(haveing, _par, negate=True)
                self.addcmd(wanting, _par)
        for key, haveing in iteritems(i_have):
            self.addcmd(haveing, _par, negate=True)