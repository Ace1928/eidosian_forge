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
class UnsetSubnet(common.NeutronUnsetCommandWithExtraArgs):
    _description = _('Unset subnet properties')

    def get_parser(self, prog_name):
        parser = super(UnsetSubnet, self).get_parser(prog_name)
        parser.add_argument('--allocation-pool', metavar='start=<ip-address>,end=<ip-address>', dest='allocation_pools', action=parseractions.MultiKeyValueAction, required_keys=['start', 'end'], help=_('Allocation pool IP addresses to be removed from this subnet e.g.: start=192.168.199.2,end=192.168.199.254 (repeat option to unset multiple allocation pools)'))
        parser.add_argument('--gateway', action='store_true', help=_('Remove gateway IP from this subnet'))
        parser.add_argument('--dns-nameserver', metavar='<dns-nameserver>', action='append', dest='dns_nameservers', help=_('DNS server to be removed from this subnet (repeat option to unset multiple DNS servers)'))
        parser.add_argument('--host-route', metavar='destination=<subnet>,gateway=<ip-address>', dest='host_routes', action=parseractions.MultiKeyValueAction, required_keys=['destination', 'gateway'], help=_('Route to be removed from this subnet e.g.: destination=10.10.0.0/16,gateway=192.168.71.254 destination: destination subnet (in CIDR notation) gateway: nexthop IP address (repeat option to unset multiple host routes)'))
        parser.add_argument('--service-type', metavar='<service-type>', action='append', dest='service_types', help=_('Service type to be removed from this subnet e.g.: network:floatingip_agent_gateway. Must be a valid device owner value for a network port (repeat option to unset multiple service types)'))
        _tag.add_tag_option_to_parser_for_unset(parser, _('subnet'))
        parser.add_argument('subnet', metavar='<subnet>', help=_('Subnet to modify (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        obj = client.find_subnet(parsed_args.subnet, ignore_missing=False)
        attrs = {}
        if parsed_args.gateway:
            attrs['gateway_ip'] = None
        if parsed_args.dns_nameservers:
            attrs['dns_nameservers'] = copy.deepcopy(obj.dns_nameservers)
            _update_arguments(attrs['dns_nameservers'], parsed_args.dns_nameservers, 'dns-nameserver')
        if parsed_args.host_routes:
            attrs['host_routes'] = copy.deepcopy(obj.host_routes)
            _update_arguments(attrs['host_routes'], convert_entries_to_nexthop(parsed_args.host_routes), 'host-route')
        if parsed_args.allocation_pools:
            attrs['allocation_pools'] = copy.deepcopy(obj.allocation_pools)
            _update_arguments(attrs['allocation_pools'], parsed_args.allocation_pools, 'allocation-pool')
        if parsed_args.service_types:
            attrs['service_types'] = copy.deepcopy(obj.service_types)
            _update_arguments(attrs['service_types'], parsed_args.service_types, 'service-type')
        attrs.update(self._parse_extra_properties(parsed_args.extra_properties))
        if attrs:
            client.update_subnet(obj, **attrs)
        _tag.update_tags_for_unset(client, obj, parsed_args)