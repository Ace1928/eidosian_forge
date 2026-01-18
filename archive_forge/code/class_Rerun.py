import logging
import os.path
from oslo_serialization import jsonutils
from osc_lib.command import command
from mistralclient.commands.v2 import base
from mistralclient.commands.v2 import executions
from mistralclient import utils
class Rerun(command.ShowOne):
    """Rerun an existing task."""

    def get_parser(self, prog_name):
        parser = super(Rerun, self).get_parser(prog_name)
        parser.add_argument('id', help='Task identifier')
        parser.add_argument('--resume', action='store_true', dest='resume', default=False, help='rerun only failed or unstarted action executions for with-items task')
        parser.add_argument('-e', '--env', dest='env', help='Environment variables')
        return parser

    def take_action(self, parsed_args):
        mistral_client = self.app.client_manager.workflow_engine
        env = utils.load_file(parsed_args.env) if parsed_args.env and os.path.isfile(parsed_args.env) else utils.load_content(parsed_args.env)
        execution = mistral_client.tasks.rerun(parsed_args.id, reset=not parsed_args.resume, env=env)
        return TaskFormatter.format(execution)