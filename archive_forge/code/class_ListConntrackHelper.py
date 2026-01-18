import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ListConntrackHelper(command.Lister):
    _description = _('List L3 conntrack helpers')

    def get_parser(self, prog_name):
        parser = super(ListConntrackHelper, self).get_parser(prog_name)
        parser.add_argument('router', metavar='<router>', help=_('Router that the conntrack helper belong to'))
        parser.add_argument('--helper', metavar='<helper>', help=_('The netfilter conntrack helper module'))
        parser.add_argument('--protocol', metavar='<protocol>', help=_('The network protocol for the netfilter conntrack target rule'))
        parser.add_argument('--port', metavar='<port>', help=_('The network port for the netfilter conntrack target rule'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        columns = ('id', 'router_id', 'helper', 'protocol', 'port')
        column_headers = ('ID', 'Router ID', 'Helper', 'Protocol', 'Port')
        attrs = _get_attrs(client, parsed_args)
        data = client.conntrack_helpers(attrs.pop('router_id'), **attrs)
        return (column_headers, (utils.get_item_properties(s, columns, formatters={}) for s in data))