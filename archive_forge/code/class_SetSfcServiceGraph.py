import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
class SetSfcServiceGraph(command.Command):
    _description = _('Set service graph properties')

    def get_parser(self, prog_name):
        parser = super(SetSfcServiceGraph, self).get_parser(prog_name)
        parser.add_argument('--name', metavar='<name>', help=_('Name of the service graph'))
        parser.add_argument('--description', metavar='<description>', help=_('Description for the service graph'))
        parser.add_argument('service_graph', metavar='<service-graph>', help=_('Service graph to modify (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        service_graph_id = client.find_sfc_service_graph(parsed_args.service_graph, ignore_missing=False)['id']
        attrs = _get_common_attrs(self.app.client_manager, parsed_args, is_create=False)
        try:
            client.update_sfc_service_graph(service_graph_id, **attrs)
        except Exception as e:
            msg = _("Failed to update service graph '%(service_graph)s': %(e)s") % {'service_graph': parsed_args.service_graph, 'e': e}
            raise exceptions.CommandError(msg)