import argparse
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.vpn import utils as vpn_utils
def _format_peer_cidrs(ipsec_site_connection):
    try:
        return '\n'.join([jsonutils.dumps(cidrs) for cidrs in ipsec_site_connection['peer_cidrs']])
    except (TypeError, KeyError):
        return ''