from osc_lib import utils as osc_utils
from osc_lib.command import command
from troveclient.i18n import _
from troveclient import utils
class ShowDatabaseQuota(command.Lister):
    _description = _('Show quotas for a project.')
    columns = ['Resource', 'In Use', 'Reserved', 'Limit']

    def get_parser(self, prog_name):
        parser = super(ShowDatabaseQuota, self).get_parser(prog_name)
        parser.add_argument('project', help=_('Id or name of the project.'))
        return parser

    def take_action(self, parsed_args):
        db_quota = self.app.client_manager.database.quota
        project_id = utils.get_project_id(self.app.client_manager.identity, parsed_args.project)
        quota = [osc_utils.get_item_properties(q, self.columns) for q in db_quota.show(project_id)]
        return (self.columns, quota)