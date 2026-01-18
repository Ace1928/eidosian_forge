from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.common import validators
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.bgp import peer as bgp_peer
def add_common_arguments(parser):
    utils.add_boolean_argument(parser, '--advertise-floating-ip-host-routes', help=_('Whether to enable or disable the advertisement of floating-ip host routes by the BGP speaker. By default floating ip host routes will be advertised by the BGP speaker.'))
    utils.add_boolean_argument(parser, '--advertise-tenant-networks', help=_('Whether to enable or disable the advertisement of tenant network routes by the BGP speaker. By default tenant network routes will be advertised by the BGP speaker.'))