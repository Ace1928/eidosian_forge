from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmpl_vrf_all(config_data):
    conf = config_data.get('vrf_all', {})
    commands = []
    if conf:
        if 'source_rt_import_policy' in conf:
            commands.append('vrf all source rt import-policy')
        if 'label_mode' in conf:
            command = 'vrf all label mode'
            if 'per_ce' in conf.get('label_mode'):
                command += ' per-ce'
            elif 'per_vrf' in conf.get('label_mode'):
                command += ' per-vrf'
            elif 'route_policy' in conf.get('label_mode'):
                command += ' route-policy ' + conf['route_policy']
        if 'table_policy' in conf:
            command = 'vrf all table-policy ' + conf['table_policy']
            commands.append(command)
        return command