import argparse
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
def _get_protocol_port(rule):
    proto = rule['protocol']
    port_min = rule['port_range_min']
    port_max = rule['port_range_max']
    if proto in ('tcp', 'udp'):
        if port_min and port_min == port_max:
            protocol_port = '%s/%s' % (port_min, proto)
        elif port_min:
            protocol_port = '%s-%s/%s' % (port_min, port_max, proto)
        else:
            protocol_port = proto
    elif proto == 'icmp':
        icmp_opts = []
        if port_min is not None:
            icmp_opts.append('type:%s' % port_min)
        if port_max is not None:
            icmp_opts.append('code:%s' % port_max)
        if icmp_opts:
            protocol_port = 'icmp (%s)' % ', '.join(icmp_opts)
        else:
            protocol_port = 'icmp'
    elif proto is not None:
        protocol_port = proto
    else:
        protocol_port = None
    return protocol_port