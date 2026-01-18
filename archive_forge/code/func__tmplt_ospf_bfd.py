from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_bfd(config_data):
    if 'bfd' in config_data:
        command = 'bfd'
        if 'minimum_interval' in config_data['bfd']:
            command += ' minimum-interval {minimum_interval}'.format(**config_data['bfd'])
        if 'multiplier' in config_data['bfd']:
            command += ' multiplier {multiplier}'.format(**config_data['bfd'])
        return command