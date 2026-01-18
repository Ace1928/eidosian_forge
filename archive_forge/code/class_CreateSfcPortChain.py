import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
class CreateSfcPortChain(command.ShowOne):
    _description = _('Create a port chain')

    def get_parser(self, prog_name):
        parser = super(CreateSfcPortChain, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<name>', help=_('Name of the port chain'))
        parser.add_argument('--description', metavar='<description>', help=_('Description for the port chain'))
        parser.add_argument('--flow-classifier', default=[], metavar='<flow-classifier>', dest='flow_classifiers', action='append', help=_('Add flow classifier (name or ID). This option can be repeated.'))
        parser.add_argument('--chain-parameters', metavar='correlation=<correlation-type>,symmetric=<boolean>', action=parseractions.MultiKeyValueAction, optional_keys=['correlation', 'symmetric'], help=_('Dictionary of chain parameters. Supports correlation=(mpls|nsh) (default is mpls) and symmetric=(true|false).'))
        parser.add_argument('--port-pair-group', metavar='<port-pair-group>', dest='port_pair_groups', required=True, action='append', help=_('Add port pair group (name or ID). This option can be repeated.'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        attrs = _get_common_attrs(self.app.client_manager, parsed_args)
        obj = client.create_sfc_port_chain(**attrs)
        display_columns, columns = utils.get_osc_show_columns_for_sdk_resource(obj, _attr_map_dict, ['location', 'tenant_id'])
        data = utils.get_dict_properties(obj, columns)
        return (display_columns, data)