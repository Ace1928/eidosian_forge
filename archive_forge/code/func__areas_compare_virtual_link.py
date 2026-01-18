from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.facts.facts import Facts
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.rm_templates.ospfv2 import (
def _areas_compare_virtual_link(self, want, have):
    wvlinks = want.get('virtual_link', {})
    hvlinks = have.get('virtual_link', {})
    for name, entry in iteritems(wvlinks):
        self._area_compare_virtual_link(want=entry, have=hvlinks.pop(name, {}))
    for name, entry in iteritems(hvlinks):
        self._area_compare_virtual_link(want={}, have=entry)