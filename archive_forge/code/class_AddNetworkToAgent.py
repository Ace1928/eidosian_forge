import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class AddNetworkToAgent(command.Command):
    _description = _('Add network to an agent')

    def get_parser(self, prog_name):
        parser = super(AddNetworkToAgent, self).get_parser(prog_name)
        parser.add_argument('--dhcp', action='store_true', help=_('Add network to a DHCP agent'))
        parser.add_argument('agent_id', metavar='<agent-id>', help=_('Agent to which a network is added (ID only)'))
        parser.add_argument('network', metavar='<network>', help=_('Network to be added to an agent (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        agent = client.get_agent(parsed_args.agent_id)
        network = client.find_network(parsed_args.network, ignore_missing=False)
        if parsed_args.dhcp:
            try:
                client.add_dhcp_agent_to_network(agent, network)
            except Exception:
                msg = 'Failed to add {} to {}'.format(network.name, agent.agent_type)
                exceptions.CommandError(msg)