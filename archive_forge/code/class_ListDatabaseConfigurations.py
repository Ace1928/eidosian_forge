import json
from osc_lib.command import command
from osc_lib import utils as osc_utils
from oslo_utils import uuidutils
from troveclient import exceptions
from troveclient.i18n import _
class ListDatabaseConfigurations(command.Lister):
    _description = _('List database configurations')
    columns = ['ID', 'Name', 'Description', 'Datastore Name', 'Datastore Version Name', 'Datastore Version Number']

    def get_parser(self, prog_name):
        parser = super(ListDatabaseConfigurations, self).get_parser(prog_name)
        parser.add_argument('--limit', dest='limit', metavar='<limit>', type=int, default=None, help=_('Limit the number of results displayed.'))
        parser.add_argument('--marker', dest='marker', metavar='<ID>', help=_('Begin displaying the results for IDs greater than the specified marker. When used with --limit, set this to the last ID displayed in the previous run.'))
        return parser

    def take_action(self, parsed_args):
        db_configurations = self.app.client_manager.database.configurations
        config = db_configurations.list(limit=parsed_args.limit, marker=parsed_args.marker)
        config = [osc_utils.get_item_properties(c, self.columns) for c in config]
        return (self.columns, config)