import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class ShowMeter(command.ShowOne):
    _description = _('Show network meter')

    def get_parser(self, prog_name):
        parser = super(ShowMeter, self).get_parser(prog_name)
        parser.add_argument('meter', metavar='<meter>', help=_('Meter to display (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        obj = client.find_metering_label(parsed_args.meter, ignore_missing=False)
        display_columns, columns = _get_columns(obj)
        data = utils.get_item_properties(obj, columns)
        return (display_columns, data)