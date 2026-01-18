from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.providers.providers import (
def _render_prefix_list_in(self, item, config=None):
    cmd = 'neighbor %s prefix-list %s in' % (item['neighbor'], item['prefix_list_in'])
    if not config or cmd not in config:
        return cmd