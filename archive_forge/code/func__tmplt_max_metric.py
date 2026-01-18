from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_max_metric(proc):
    max_metric = proc['max_metric']
    command = 'max-metric router-lsa'
    if max_metric.get('router_lsa', {}).get('set') is False:
        command = 'no {0}'.format(command)
    else:
        external_lsa = max_metric.get('router_lsa', {}).get('external_lsa', {})
        include_stub = max_metric.get('router_lsa', {}).get('include_stub', {})
        on_startup = max_metric.get('router_lsa', {}).get('on_startup', {})
        summary_lsa = max_metric.get('router_lsa', {}).get('summary_lsa', {})
        if external_lsa:
            command += ' external-lsa'
            if external_lsa.get('max_metric_value'):
                command += ' {max_metric_value}'.format(**external_lsa)
        if include_stub:
            command += ' include-stub'
        if on_startup:
            command += ' on-startup'
            if on_startup.get('wait_period'):
                command += ' {wait_period}'.format(**on_startup)
            if on_startup.get('wait_for_bgp_asn'):
                command += ' wait-for bgp {wait_for_bgp_asn}'.format(**on_startup)
        if summary_lsa:
            command += ' summary-lsa'
            if summary_lsa.get('max_metric_value'):
                command += ' {max_metric_value}'.format(**summary_lsa)
    return command