from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_utils import uuidutils
from troveclient.i18n import _
from troveclient.osc.v1 import base
from troveclient import utils as trove_utils
class ListDatabaseBackups(command.Lister):
    _description = _('List database backups')
    columns = ['ID', 'Instance ID', 'Name', 'Status', 'Parent ID', 'Updated', 'Project ID']

    def get_parser(self, prog_name):
        parser = super(ListDatabaseBackups, self).get_parser(prog_name)
        parser.add_argument('--limit', dest='limit', metavar='<limit>', default=None, help=_('Return up to N number of the most recent bcakups.'))
        parser.add_argument('--marker', dest='marker', metavar='<ID>', type=str, default=None, help=_('Begin displaying the results for IDs greater than thespecified marker. When used with ``--limit``, set this to the last ID displayed in the previous run.'))
        parser.add_argument('--datastore', dest='datastore', metavar='<datastore>', default=None, help=_('ID or name of the datastore (to filter backups by).'))
        parser.add_argument('--instance-id', default=None, help=_('Filter backups by database instance ID. Deprecated since Xena. Use -i/--instance instead.'))
        parser.add_argument('-i', '--instance', default=None, help=_('Filter backups by database instance(ID or name).'))
        parser.add_argument('--all-projects', action='store_true', help=_('Get all the backups of all the projects(Admin only).'))
        parser.add_argument('--project-id', default=None, help=_('Filter backups by project ID.'))
        return parser

    def take_action(self, parsed_args):
        database_backups = self.app.client_manager.database.backups
        instance_id = parsed_args.instance or parsed_args.instance_id
        if instance_id:
            instance_mgr = self.app.client_manager.database.instances
            instance_id = trove_utils.get_resource_id(instance_mgr, instance_id)
        items = database_backups.list(limit=parsed_args.limit, datastore=parsed_args.datastore, marker=parsed_args.marker, instance_id=instance_id, all_projects=parsed_args.all_projects, project_id=parsed_args.project_id)
        backups = items
        while items.next and (not parsed_args.limit):
            items = database_backups.list(marker=items.next, datastore=parsed_args.datastore, instance_id=parsed_args.instance_id, all_projects=parsed_args.all_projects, project_id=parsed_args.project_id)
            backups += items
        backups = [osc_utils.get_item_properties(b, self.columns) for b in backups]
        return (self.columns, backups)