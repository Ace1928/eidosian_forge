from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.qos import rule as qos_rule
def add_dscp_marking_arguments(parser):
    parser.add_argument('--dscp-mark', required=True, type=str, help=_('DSCP mark: value can be 0, even numbers from 8-56,                 excluding 42, 44, 50, 52, and 54.'))