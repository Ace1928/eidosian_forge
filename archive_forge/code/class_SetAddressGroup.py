import logging
import netaddr
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class SetAddressGroup(common.NeutronCommandWithExtraArgs):
    _description = _('Set address group properties')

    def get_parser(self, prog_name):
        parser = super(SetAddressGroup, self).get_parser(prog_name)
        parser.add_argument('address_group', metavar='<address-group>', help=_('Address group to modify (name or ID)'))
        parser.add_argument('--name', metavar='<name>', help=_('Set address group name'))
        parser.add_argument('--description', metavar='<description>', help=_('Set address group description'))
        parser.add_argument('--address', metavar='<ip-address>', action='append', default=[], help=_('IP address or CIDR (repeat option to set multiple addresses)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        obj = client.find_address_group(parsed_args.address_group, ignore_missing=False)
        attrs = {}
        if parsed_args.name is not None:
            attrs['name'] = parsed_args.name
        if parsed_args.description is not None:
            attrs['description'] = parsed_args.description
        attrs.update(self._parse_extra_properties(parsed_args.extra_properties))
        if attrs:
            client.update_address_group(obj, **attrs)
        if parsed_args.address:
            client.add_addresses_to_address_group(obj, _format_addresses(parsed_args.address))