from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.frr.frr.plugins.module_utils.network.frr.providers.providers import (
def _negate_config(self, config, safe_list=None):
    commands = list()
    matches = re.findall('(neighbor \\S+)', config, re.M)
    for item in set(matches).difference(safe_list):
        commands.append('no %s' % item)
    return commands