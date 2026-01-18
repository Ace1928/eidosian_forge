import itertools
import logging
from cliff import command
from cliff import show
from designateclient.v2.cli import common
class ResetQuotasCommand(command.Command):
    """Reset quotas"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        common.add_all_common_options(parser)
        parser.add_argument('--project-id', help='Project ID')
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.dns
        common.set_all_common_headers(client, parsed_args)
        proj_id = parsed_args.project_id or client.session.get_project_id()
        if parsed_args.project_id != client.session.get_project_id():
            common.set_all_projects(client, True)
        client.quotas.reset(proj_id)
        LOG.info('Quota for project %s was reset', parsed_args.project_id)