from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.arista.eos.plugins.module_utils.network.eos.providers.providers import (
def _render_graceful_restart(self, item, config=None):
    cmd = 'neighbor %s graceful-restart' % item['neighbor']
    if item['graceful_restart'] is False:
        cmd = 'no ' + cmd
    if config:
        config_el = [x.strip() for x in config.split('\n')]
        if cmd in config_el:
            return
    return cmd