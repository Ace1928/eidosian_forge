from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_protocol_shutdown(config_data):
    if 'protocol_shutdown' in config_data:
        if 'set' in config_data['protocol_shutdown']:
            command = 'protocol-shutdown'
        if 'host_mode' in config_data['protocol_shutdown']:
            command = 'protocol-shutdown host-mode'
        if 'on_reload' in config_data['protocol_shutdown']:
            command = 'protocol-shutdown on-reload'
        return command