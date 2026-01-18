import copy
import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import tags as _tag
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
def _get_common_parse_arguments(parser, is_create=True):
    parser.add_argument('--allocation-pool', metavar='start=<ip-address>,end=<ip-address>', dest='allocation_pools', action=parseractions.MultiKeyValueAction, required_keys=['start', 'end'], help=_('Allocation pool IP addresses for this subnet e.g.: start=192.168.199.2,end=192.168.199.254 (repeat option to add multiple IP addresses)'))
    if not is_create:
        parser.add_argument('--no-allocation-pool', action='store_true', help=_('Clear associated allocation-pools from the subnet. Specify both --allocation-pool and --no-allocation-pool to overwrite the current allocation pool information.'))
    parser.add_argument('--dns-nameserver', metavar='<dns-nameserver>', action='append', dest='dns_nameservers', help=_('DNS server for this subnet (repeat option to set multiple DNS servers)'))
    if not is_create:
        parser.add_argument('--no-dns-nameservers', action='store_true', help=_('Clear existing information of DNS Nameservers. Specify both --dns-nameserver and --no-dns-nameserver to overwrite the current DNS Nameserver information.'))
    parser.add_argument('--host-route', metavar='destination=<subnet>,gateway=<ip-address>', dest='host_routes', action=parseractions.MultiKeyValueAction, required_keys=['destination', 'gateway'], help=_('Additional route for this subnet e.g.: destination=10.10.0.0/16,gateway=192.168.71.254 destination: destination subnet (in CIDR notation) gateway: nexthop IP address (repeat option to add multiple routes)'))
    if not is_create:
        parser.add_argument('--no-host-route', action='store_true', help=_('Clear associated host-routes from the subnet. Specify both --host-route and --no-host-route to overwrite the current host route information.'))
    parser.add_argument('--service-type', metavar='<service-type>', action='append', dest='service_types', help=_('Service type for this subnet e.g.: network:floatingip_agent_gateway. Must be a valid device owner value for a network port (repeat option to set multiple service types)'))