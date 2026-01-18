import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
class ListSfcPortChain(command.Lister):
    _description = _('List port chains')

    def get_parser(self, prog_name):
        parser = super(ListSfcPortChain, self).get_parser(prog_name)
        parser.add_argument('--long', action='store_true', default=False, help=_('List additional fields in output'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        data = client.sfc_port_chains()
        headers, columns = column_util.get_column_definitions(_attr_map, long_listing=parsed_args.long)
        return (headers, (utils.get_dict_properties(s, columns) for s in data))