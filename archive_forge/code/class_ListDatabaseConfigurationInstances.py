import json
from osc_lib.command import command
from osc_lib import utils as osc_utils
from oslo_utils import uuidutils
from troveclient import exceptions
from troveclient.i18n import _
class ListDatabaseConfigurationInstances(command.Lister):
    _description = _('Lists all instances associated with a configuration group.')
    columns = ['ID', 'Name']

    def get_parser(self, prog_name):
        parser = super(ListDatabaseConfigurationInstances, self).get_parser(prog_name)
        parser.add_argument('configuration_group', metavar='<configuration_group>', help=_('ID or name of the configuration group.'))
        parser.add_argument('--limit', metavar='<limit>', default=None, type=int, help=_('Limit the number of results displayed.'))
        parser.add_argument('--marker', metavar='<ID>', default=None, type=str, help=_('Begin displaying the results for IDs greater than the specified marker. When used with --limit, set this to the last ID displayed in the previous run.'))
        return parser

    def take_action(self, parsed_args):
        db_configurations = self.app.client_manager.database.configurations
        configuration = osc_utils.find_resource(db_configurations, parsed_args.configuration_group)
        params = db_configurations.instances(configuration, limit=parsed_args.limit, marker=parsed_args.marker)
        instance = [osc_utils.get_item_properties(p, self.columns) for p in params]
        return (self.columns, instance)