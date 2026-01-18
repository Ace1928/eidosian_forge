import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class ShowMeterRule(command.ShowOne):
    _description = _('Display meter rules details')

    def get_parser(self, prog_name):
        parser = super(ShowMeterRule, self).get_parser(prog_name)
        parser.add_argument('meter_rule_id', metavar='<meter-rule-id>', help=_('Meter rule (ID only)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        obj = client.find_metering_label_rule(parsed_args.meter_rule_id, ignore_missing=False)
        display_columns, columns = _get_columns(obj)
        data = utils.get_item_properties(obj, columns)
        return (display_columns, data)