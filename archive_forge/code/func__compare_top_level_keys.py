from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.static_routes import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.utils.utils import (
def _compare_top_level_keys(self, want, have):
    if self.state == 'deleted' and have:
        _have = {}
        for addf in ['ipv4', 'ipv6']:
            _temp_sr = {}
            for k, ha in iteritems(have.get(addf, {})):
                if k in want.get(addf, {}):
                    _temp_sr[k] = ha
                if _temp_sr:
                    _have[addf] = _temp_sr
        if _have:
            have = _have
            want = {}
    if self.state != 'deleted':
        for _afi, routes in want.items():
            self._compare(s_want=routes, s_have=have.pop(_afi, {}), afi=_afi)
    if self.state in ['overridden', 'deleted']:
        for _afi, routes in have.items():
            self._compare(s_want={}, s_have=routes, afi=_afi)