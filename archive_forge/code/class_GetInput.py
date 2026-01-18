import logging
from oslo_serialization import jsonutils
from osc_lib.command import command
from mistralclient.commands.v2 import base
from mistralclient import utils
class GetInput(command.Command):
    """Show Action execution input data."""

    def get_parser(self, prog_name):
        parser = super(GetInput, self).get_parser(prog_name)
        parser.add_argument('id', help='Action execution ID.')
        return parser

    def take_action(self, parsed_args):
        mistral_client = self.app.client_manager.workflow_engine
        result = mistral_client.action_executions.get(parsed_args.id).input
        try:
            result = jsonutils.loads(result)
            result = jsonutils.dumps(result, indent=4) + '\n'
        except Exception:
            LOG.debug('Task result is not JSON.')
        self.app.stdout.write(result or '\n')