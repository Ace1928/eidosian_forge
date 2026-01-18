import logging
import netaddr
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class ShowAddressGroup(command.ShowOne):
    _description = _('Display address group details')

    def get_parser(self, prog_name):
        parser = super(ShowAddressGroup, self).get_parser(prog_name)
        parser.add_argument('address_group', metavar='<address-group>', help=_('Address group to display (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        obj = client.find_address_group(parsed_args.address_group, ignore_missing=False)
        display_columns, columns = _get_columns(obj)
        data = utils.get_item_properties(obj, columns, formatters={})
        return (display_columns, data)