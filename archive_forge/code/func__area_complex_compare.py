from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.ospfv2 import (
def _area_complex_compare(self, want, have, area_id):
    area_complex_parsers = ['filter_list', 'ranges']
    for _parser in area_complex_parsers:
        wantr = want.get(_parser, {})
        haver = have.get(_parser, {})
        for key, wanting in iteritems(wantr):
            haveing = have.pop(key, {})
            haveing['area_id'] = area_id
            wanting['area_id'] = area_id
            if wanting != haveing:
                if haveing and self.state in ['overridden', 'replaced']:
                    self.addcmd(haveing, _parser, negate=True)
                self.addcmd(wanting, _parser, False)
        for key, haveing in iteritems(haver):
            haveing['area_id'] = area_id
            self.addcmd(haveing, _parser, negate=True)