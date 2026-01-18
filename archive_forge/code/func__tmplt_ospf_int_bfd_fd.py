from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_int_bfd_fd(config_data):
    command = _compute_command(config_data)
    bfd = config_data['address_family']['bfd']
    if bfd.get('fast_detect') and bfd['fast_detect'].get('set'):
        command += ' bfd fast-detect'
    elif bfd.get('fast_detect') and bfd['fast_detect'].get('disable'):
        command += ' bfd fast-detect disable'
    elif bfd.get('fast_detect') and bfd['fast_detect'].get('strict_mode'):
        command += ' bfd fast-detect strict-mode'
    return command