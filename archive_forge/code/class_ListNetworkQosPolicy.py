import logging
from cliff import columns as cliff_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class ListNetworkQosPolicy(command.Lister):
    _description = _('List QoS policies')

    def get_parser(self, prog_name):
        parser = super(ListNetworkQosPolicy, self).get_parser(prog_name)
        parser.add_argument('--project', metavar='<project>', help=_('List qos policies according to their project (name or ID)'))
        identity_common.add_project_domain_option_to_parser(parser)
        shared_group = parser.add_mutually_exclusive_group()
        shared_group.add_argument('--share', action='store_true', help=_('List qos policies shared between projects'))
        shared_group.add_argument('--no-share', action='store_true', help=_('List qos policies not shared between projects'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        columns = ('id', 'name', 'is_shared', 'is_default', 'project_id')
        column_headers = ('ID', 'Name', 'Shared', 'Default', 'Project')
        attrs = _get_attrs(self.app.client_manager, parsed_args)
        data = client.qos_policies(**attrs)
        return (column_headers, (utils.get_item_properties(s, columns, formatters={}) for s in data))