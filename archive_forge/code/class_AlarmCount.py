from cliff import lister
from cliff import show
from vitrageclient.common import utils
from vitrageclient import exceptions as exc
class AlarmCount(show.ShowOne):
    """Show a count of all alarms"""

    def get_parser(self, prog_name):
        parser = super(AlarmCount, self).get_parser(prog_name)
        parser.add_argument('--all-tenants', default=False, dest='all_tenants', action='store_true', help='Shows counts for alarms of all the tenants')
        return parser

    @property
    def formatter_default(self):
        return 'json'

    def take_action(self, parsed_args):
        all_tenants = parsed_args.all_tenants
        counts = utils.get_client(self).alarm.count(all_tenants=all_tenants)
        return self.dict2columns(counts)