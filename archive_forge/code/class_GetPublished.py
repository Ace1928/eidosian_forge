import logging
import os.path
from oslo_serialization import jsonutils
from osc_lib.command import command
from mistralclient.commands.v2 import base
from mistralclient.commands.v2 import executions
from mistralclient import utils
class GetPublished(command.Command):
    """Show task published variables."""

    def get_parser(self, prog_name):
        parser = super(GetPublished, self).get_parser(prog_name)
        parser.add_argument('id', help='Task ID')
        return parser

    def take_action(self, parsed_args):
        mistral_client = self.app.client_manager.workflow_engine
        res = mistral_client.tasks.get(parsed_args.id)
        published = res.published
        published_glob = res.published_global if hasattr(res, 'published_global') else None
        try:
            published = jsonutils.loads(published)
            published = jsonutils.dumps(published, indent=4) + '\n'
            if published_glob:
                published_glob = jsonutils.loads(published_glob)
                published += jsonutils.dumps(published_glob, indent=4) + '\n'
        except Exception:
            LOG.debug('Task result is not JSON.')
        self.app.stdout.write(published or '\n')