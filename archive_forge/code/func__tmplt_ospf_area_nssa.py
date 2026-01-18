from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_area_nssa(config_data):
    if 'nssa' in config_data:
        nssa_data = config_data['nssa']
        command = 'area {area_id} nssa'.format(**config_data)
        if 'default_information_originate' in nssa_data:
            default_info = nssa_data['default_information_originate']
            command += ' default-information-originate'
            metric = default_info.get('metric')
            if metric is not None:
                command += ' metric {metric}'.format(metric=metric)
            metric_type = default_info.get('metric_type')
            if metric_type is not None:
                command += ' metric-type {metric_type}'.format(metric_type=metric_type)
            if default_info.get('nssa_only'):
                command += ' nssa-only'
        if nssa_data.get('no_ext_capability'):
            command += ' no-ext-capability'
        if nssa_data.get('no_redistribution'):
            command += ' no-redistribution'
        if nssa_data.get('no_summary'):
            command += ' no-summary'
        return command