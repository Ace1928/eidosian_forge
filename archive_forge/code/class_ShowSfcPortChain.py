import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
class ShowSfcPortChain(command.ShowOne):
    _description = _('Display port chain details')

    def get_parser(self, prog_name):
        parser = super(ShowSfcPortChain, self).get_parser(prog_name)
        parser.add_argument('port_chain', metavar='<port-chain>', help=_('Port chain to display (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        pc_id = client.find_sfc_port_chain(parsed_args.port_chain, ignore_missing=False)['id']
        obj = client.get_sfc_port_chain(pc_id)
        display_columns, columns = utils.get_osc_show_columns_for_sdk_resource(obj, _attr_map_dict, ['location', 'tenant_id'])
        data = utils.get_dict_properties(obj, columns)
        return (display_columns, data)