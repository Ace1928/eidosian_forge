from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class ShowIPAvailability(command.ShowOne):
    _description = _('Show network IP availability details')

    def get_parser(self, prog_name):
        parser = super(ShowIPAvailability, self).get_parser(prog_name)
        parser.add_argument('network', metavar='<network>', help=_('Show IP availability for a specific network (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        network_id = client.find_network(parsed_args.network, ignore_missing=False).id
        obj = client.find_network_ip_availability(network_id, ignore_missing=False)
        display_columns, columns = _get_columns(obj)
        data = utils.get_item_properties(obj, columns, formatters=_formatters)
        return (display_columns, data)