import logging
from oslo_serialization import jsonutils
from osc_lib.command import command
from mistralclient.commands.v2 import base
from mistralclient import utils
class GetOutput(command.Command):
    """Show Action execution output data."""

    def get_parser(self, prog_name):
        parser = super(GetOutput, self).get_parser(prog_name)
        parser.add_argument('id', help='Action execution ID.')
        return parser

    def take_action(self, parsed_args):
        mistral_client = self.app.client_manager.workflow_engine
        output = mistral_client.action_executions.get(parsed_args.id).output
        try:
            output = jsonutils.loads(output)
            output = jsonutils.dumps(output, indent=4) + '\n'
        except Exception:
            LOG.debug('Task result is not JSON.')
        self.app.stdout.write(output or '\n')