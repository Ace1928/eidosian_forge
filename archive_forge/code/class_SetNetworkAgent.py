import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class SetNetworkAgent(command.Command):
    _description = _('Set network agent properties')

    def get_parser(self, prog_name):
        parser = super(SetNetworkAgent, self).get_parser(prog_name)
        parser.add_argument('network_agent', metavar='<network-agent>', help=_('Network agent to modify (ID only)'))
        parser.add_argument('--description', metavar='<description>', help=_('Set network agent description'))
        admin_group = parser.add_mutually_exclusive_group()
        admin_group.add_argument('--enable', action='store_true', help=_('Enable network agent'))
        admin_group.add_argument('--disable', action='store_true', help=_('Disable network agent'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        obj = client.get_agent(parsed_args.network_agent)
        attrs = {}
        if parsed_args.description is not None:
            attrs['description'] = parsed_args.description
        if parsed_args.enable:
            attrs['is_admin_state_up'] = True
            attrs['admin_state_up'] = True
        if parsed_args.disable:
            attrs['is_admin_state_up'] = False
            attrs['admin_state_up'] = False
        client.update_agent(obj, **attrs)