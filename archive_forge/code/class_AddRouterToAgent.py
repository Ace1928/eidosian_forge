import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class AddRouterToAgent(command.Command):
    _description = _('Add router to an agent')

    def get_parser(self, prog_name):
        parser = super(AddRouterToAgent, self).get_parser(prog_name)
        parser.add_argument('--l3', action='store_true', help=_('Add router to an L3 agent'))
        parser.add_argument('agent_id', metavar='<agent-id>', help=_('Agent to which a router is added (ID only)'))
        parser.add_argument('router', metavar='<router>', help=_('Router to be added to an agent (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        agent = client.get_agent(parsed_args.agent_id)
        router = client.find_router(parsed_args.router, ignore_missing=False)
        if parsed_args.l3:
            client.add_router_to_agent(agent, router)