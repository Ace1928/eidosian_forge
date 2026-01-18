from cliff import lister
from cliff import show
from vitrageclient.common import utils
from vitrageclient import exceptions as exc
class AlarmHistory(lister.Lister):
    """List the alarm history"""

    def get_parser(self, prog_name):
        parser = super(AlarmHistory, self).get_parser(prog_name)
        parser.add_argument('--all-tenants', default=False, dest='all_tenants', action='store_true', help='Shows alarms of all the tenants in the entity graph')
        parser.add_argument('--limit', dest='limit', help='Maximal number of alarms to show. Default is 1000')
        parser.add_argument('--marker', dest='marker', help='Marker for the next page')
        parser.add_argument('--start', dest='start', help='list alarm from this date')
        parser.add_argument('--end', dest='end', help='list alarm until this date')
        return parser

    def take_action(self, parsed_args):
        all_tenants = parsed_args.all_tenants
        limit = parsed_args.limit
        marker = parsed_args.marker
        start = parsed_args.start
        end = parsed_args.end
        if end and (not start):
            raise exc.CommandError('--end argument must be used with --start')
        alarms = utils.get_client(self).alarm.history(limit=limit, marker=marker, start=start, end=end, all_tenants=all_tenants)
        return utils.list2cols_with_rename((('ID', 'vitrage_id'), ('Type', 'vitrage_type'), ('Name', 'name'), ('Resource Type', 'vitrage_resource_type'), ('Resource ID', 'vitrage_resource_id'), ('Severity', 'vitrage_operational_severity'), ('Start Time', 'start_timestamp'), ('End Time', 'end_timestamp')), alarms)