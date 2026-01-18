from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.ospfv2 import (
def _complex_compare(self, want, have):
    complex_parsers = ['distribute_list.acls', 'network']
    for _parser in complex_parsers:
        if _parser == 'distribute_list.acls':
            wdist = want.get('distribute_list', {}).get('acls', {})
            hdist = have.get('distribute_list', {}).get('acls', {})
        else:
            wdist = want.get(_parser, {})
            hdist = have.get(_parser, {})
        for key, wanting in iteritems(wdist):
            haveing = hdist.pop(key, {})
            if wanting != haveing:
                if haveing and self.state in ['overridden', 'replaced']:
                    self.addcmd(haveing, _parser, negate=True)
                self.addcmd(wanting, _parser, False)
        for key, haveing in iteritems(hdist):
            self.addcmd(haveing, _parser, negate=True)