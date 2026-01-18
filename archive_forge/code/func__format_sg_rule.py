import argparse
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
def _format_sg_rule(rule):
    formatted = []
    for field in ['direction', 'ethertype', ('protocol_port', _get_protocol_port), 'remote_ip_prefix', 'remote_group_id']:
        if isinstance(field, tuple):
            field, get_method = field
            data = get_method(rule)
        else:
            data = rule[field]
        if not data:
            continue
        if field in ('remote_ip_prefix', 'remote_group_id'):
            data = '%s: %s' % (field, data)
        formatted.append(data)
    return ', '.join(formatted)