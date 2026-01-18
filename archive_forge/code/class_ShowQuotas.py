from magnumclient.common import cliutils as utils
from magnumclient.common import utils as magnum_utils
from magnumclient.i18n import _
from osc_lib.command import command
class ShowQuotas(command.Command):
    _description = _('Show details about the given project resource quota.')

    def get_parser(self, prog_name):
        parser = super(ShowQuotas, self).get_parser(prog_name)
        parser.add_argument('--project-id', required=True, metavar='<project-id>', help='Project ID')
        parser.add_argument('--resource', required=True, metavar='<resource>', help='Resource name.')
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        mag_client = self.app.client_manager.container_infra
        project_id = parsed_args.project_id
        resource = parsed_args.resource
        quota = mag_client.quotas.get(project_id, resource)
        _show_quota(quota)