from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_utils import uuidutils
from troveclient.i18n import _
from troveclient.osc.v1 import base
from troveclient import utils as trove_utils
class ListDatabaseInstanceBackups(command.Lister):
    _description = _('Lists available backups for an instance.')
    columns = ['ID', 'Instance ID', 'Name', 'Status', 'Parent ID', 'Updated']

    def get_parser(self, prog_name):
        parser = super(ListDatabaseInstanceBackups, self).get_parser(prog_name)
        parser.add_argument('instance', metavar='<instance>', help=_('ID or name of the instance.'))
        parser.add_argument('--limit', dest='limit', metavar='<limit>', default=None, help=_('Return up to N number of the most recent bcakups.'))
        parser.add_argument('--marker', dest='marker', metavar='<ID>', type=str, default=None, help=_('Begin displaying the results for IDs greater than thespecified marker. When used with ``--limit``, set this to the last ID displayed in the previous run.'))
        return parser

    def take_action(self, parsed_args):
        database_instances = self.app.client_manager.database.instances
        instance = osc_utils.find_resource(database_instances, parsed_args.instance)
        items = database_instances.backups(instance, limit=parsed_args.limit, marker=parsed_args.marker)
        backups = items
        while items.next and (not parsed_args.limit):
            items = database_instances.backups(instance, marker=items.next)
            backups += items
        backups = [osc_utils.get_item_properties(b, self.columns) for b in backups]
        return (self.columns, backups)