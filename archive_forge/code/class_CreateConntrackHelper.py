import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class CreateConntrackHelper(command.ShowOne):
    _description = _('Create a new L3 conntrack helper')

    def get_parser(self, prog_name):
        parser = super(CreateConntrackHelper, self).get_parser(prog_name)
        parser.add_argument('router', metavar='<router>', help=_('Router for which conntrack helper will be created'))
        parser.add_argument('--helper', required=True, metavar='<helper>', help=_('The netfilter conntrack helper module'))
        parser.add_argument('--protocol', required=True, metavar='<protocol>', help=_('The network protocol for the netfilter conntrack target rule'))
        parser.add_argument('--port', required=True, metavar='<port>', type=int, help=_('The network port for the netfilter conntrack target rule'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        attrs = _get_attrs(client, parsed_args)
        obj = client.create_conntrack_helper(attrs.pop('router_id'), **attrs)
        display_columns, columns = _get_columns(obj)
        data = utils.get_item_properties(obj, columns, formatters={})
        return (display_columns, data)