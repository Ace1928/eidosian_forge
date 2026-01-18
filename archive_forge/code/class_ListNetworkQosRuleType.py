from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
class ListNetworkQosRuleType(command.Lister):
    _description = _('List QoS rule types')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        supported = parser.add_mutually_exclusive_group()
        supported.add_argument('--all-supported', action='store_true', help=_('List all the QoS rule types supported by any loaded mechanism drivers (the union of all sets of supported rules)'))
        supported.add_argument('--all-rules', action='store_true', help=_('List all QoS rule types implemented in Neutron QoS driver'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        columns = ('type',)
        column_headers = ('Type',)
        args = {}
        if parsed_args.all_supported:
            args['all_supported'] = True
        elif parsed_args.all_rules:
            args['all_rules'] = True
        data = client.qos_rule_types(**args)
        return (column_headers, (utils.get_item_properties(s, columns, formatters={}) for s in data))