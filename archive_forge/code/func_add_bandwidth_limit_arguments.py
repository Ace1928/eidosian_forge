from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.qos import rule as qos_rule
def add_bandwidth_limit_arguments(parser):
    parser.add_argument('--max-kbps', help=_('Maximum bandwidth in kbps.'))
    parser.add_argument('--max-burst-kbps', help=_('Maximum burst bandwidth in kbps.'))