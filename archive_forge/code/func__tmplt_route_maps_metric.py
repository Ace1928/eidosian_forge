from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_route_maps_metric(config_data):
    config_data = config_data['entries']['set']['metric']
    command = 'set metric'
    if config_data.get('value'):
        command += ' ' + config_data['value']
    if config_data.get('add'):
        command += ' +' + config_data['add']
    if config_data.get('igp_param'):
        command += ' ' + config_data['igp_param']
    return command