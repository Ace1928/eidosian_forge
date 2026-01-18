from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_passive_interfaces(config_data):
    if 'passive_interfaces' in config_data:
        if config_data['passive_interfaces'].get('interface'):
            if config_data['passive_interfaces'].get('set_interface'):
                for each in config_data['passive_interfaces']['interface']:
                    cmd = 'passive-interface {0}'.format(each)
            elif not config_data['passive_interfaces'].get('set_interface'):
                for each in config_data['passive_interfaces']['interface']:
                    cmd = 'no passive-interface {0}'.format(each)
        return cmd