from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
def add_common_args(parser):
    parser.add_argument('--name', help=_('Name for the firewall.'))
    parser.add_argument('--description', help=_('Description for the firewall.'))
    router = parser.add_mutually_exclusive_group()
    router.add_argument('--router', dest='routers', metavar='ROUTER', action='append', help=_('ID or name of the router associated with the firewall (requires FWaaS router insertion extension to be enabled). This option can be repeated.'))
    router.add_argument('--no-routers', action='store_true', help=_('Associate no routers with the firewall (requires FWaaS router insertion extension).'))