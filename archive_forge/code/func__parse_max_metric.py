from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
def _parse_max_metric(max_metric_dict):
    metric_cmd = 'max-metric router-lsa '
    for k, v in max_metric_dict['router_lsa'].items():
        if not v:
            continue
        if k == 'include_stub' and v:
            metric_cmd = metric_cmd + ' include-stub'
        elif k == 'on_startup':
            metric_cmd = metric_cmd + ' on-startup ' + str(v['wait_period'])
        elif k in ['summary_lsa', 'external_lsa']:
            k = re.sub('_', '-', k)
            if v.get('set'):
                metric_cmd = metric_cmd + ' ' + k
            else:
                metric_cmd = metric_cmd + ' ' + k + ' ' + str(v.get('max_metric_value'))
    return metric_cmd