import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
class ShowSfcServiceGraph(command.ShowOne):
    """Show information of a given service graph."""

    def get_parser(self, prog_name):
        parser = super(ShowSfcServiceGraph, self).get_parser(prog_name)
        parser.add_argument('service_graph', metavar='<service-graph>', help=_('ID or name of the service graph to display.'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        sg_id = client.find_sfc_service_graph(parsed_args.service_graph, ignore_missing=False)['id']
        obj = client.get_sfc_service_graph(sg_id)
        display_columns, columns = utils.get_osc_show_columns_for_sdk_resource(obj, _attr_map_dict, ['location', 'tenant_id'])
        data = utils.get_dict_properties(obj, columns)
        return (display_columns, data)