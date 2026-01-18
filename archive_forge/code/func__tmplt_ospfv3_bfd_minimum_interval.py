from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospfv3_bfd_minimum_interval(config_data):
    if 'bfd' in config_data:
        if 'minimum_interval' in config_data['bfd']:
            command = 'bfd minimum-interval {minimum_interval}'.format(**config_data['bfd'])
            return command