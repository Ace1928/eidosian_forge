from osc_lib import utils as osc_utils
from osc_lib.command import command
from troveclient.i18n import _
from troveclient import utils
class UpdateDatabaseQuota(command.ShowOne):
    _description = _('Update quotas for a project.')

    def get_parser(self, prog_name):
        parser = super(UpdateDatabaseQuota, self).get_parser(prog_name)
        parser.add_argument('project', help=_('Id or name of the project.'))
        parser.add_argument('resource', metavar='<resource>', help=_('Resource name.'))
        parser.add_argument('limit', metavar='<limit>', type=int, help=_('New limit to set for the named resource.'))
        return parser

    def take_action(self, parsed_args):
        db_quota = self.app.client_manager.database.quota
        project_id = utils.get_project_id(self.app.client_manager.identity, parsed_args.project)
        update_params = {parsed_args.resource: parsed_args.limit}
        updated_quota = db_quota.update(project_id, update_params)
        return zip(*sorted(updated_quota.items()))