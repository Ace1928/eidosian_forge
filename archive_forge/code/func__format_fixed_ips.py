import argparse
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import dns
from neutronclient.neutron.v2_0.qos import policy as qos_policy
def _format_fixed_ips(port):
    try:
        return '\n'.join([jsonutils.dumps(ip) for ip in port['fixed_ips']])
    except (TypeError, KeyError):
        return ''