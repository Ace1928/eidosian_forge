from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.frr.frr.plugins.module_utils.network.frr.providers.providers import (
def _render_local_as(self, item, config=None):
    cmd = 'neighbor %s local-as %s' % (item['neighbor'], item['local_as'])
    if not config or cmd not in config:
        return cmd