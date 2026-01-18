from cliff import lister
from cliff import show
from vitrageclient.common import utils
from vitrageclient import exceptions as exc
class AlarmList(lister.Lister):
    """List all alarms"""

    def get_parser(self, prog_name):
        parser = super(AlarmList, self).get_parser(prog_name)
        parser.add_argument('vitrage_id', default='all', nargs='?', metavar='<vitrage id>', help='Vitrage id of the affected resource')
        parser.add_argument('--all-tenants', default=False, dest='all_tenants', action='store_true', help='Shows alarms of all the tenants in the entity graph')
        parser.add_argument('--limit', dest='limit', help='Maximal number of alarms to show. Default is 1000')
        parser.add_argument('--marker', dest='marker', help='Marker for the next page')
        return parser

    def take_action(self, parsed_args):
        vitrage_id = parsed_args.vitrage_id
        all_tenants = parsed_args.all_tenants
        limit = parsed_args.limit
        marker = parsed_args.marker
        alarms = utils.get_client(self).alarm.list(vitrage_id=vitrage_id, limit=limit, marker=marker, all_tenants=all_tenants)
        return utils.list2cols_with_rename((('ID', 'vitrage_id'), ('Type', 'vitrage_type'), ('Name', 'name'), ('Resource Type', 'vitrage_resource_type'), ('Resource ID', 'vitrage_resource_id'), ('Severity', 'vitrage_operational_severity'), ('Update Time', 'update_timestamp')), alarms)