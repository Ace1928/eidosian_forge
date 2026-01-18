import logging
import netaddr
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class UnsetAddressGroup(command.Command):
    _description = _('Unset address group properties')

    def get_parser(self, prog_name):
        parser = super(UnsetAddressGroup, self).get_parser(prog_name)
        parser.add_argument('address_group', metavar='<address-group>', help=_('Address group to modify (name or ID)'))
        parser.add_argument('--address', metavar='<ip-address>', action='append', default=[], help=_('IP address or CIDR (repeat option to unset multiple addresses)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        obj = client.find_address_group(parsed_args.address_group, ignore_missing=False)
        if parsed_args.address:
            client.remove_addresses_from_address_group(obj, _format_addresses(parsed_args.address))