from cliff import lister
from cliff import show
from vitrageclient.common import utils
class ResourceCount(show.ShowOne):
    """Show a count of all resources"""

    def get_parser(self, prog_name):
        parser = super(ResourceCount, self).get_parser(prog_name)
        parser.add_argument('--type', dest='resource_type', metavar='<resource type>', help='Type of resource')
        parser.add_argument('--all-tenants', default=False, dest='all_tenants', action='store_true', help='Shows resources of all the tenants')
        (parser.add_argument('--filter', metavar='<query>', help='resource query'),)
        (parser.add_argument('--group-by', dest='group_by', metavar='<group_by>', default='vitrage_type', help="A resource data field, to group by it's values"),)
        return parser

    @property
    def formatter_default(self):
        return 'json'

    def take_action(self, parsed_args):
        resource_type = parsed_args.resource_type
        all_tenants = parsed_args.all_tenants
        query = parsed_args.filter
        group_by = parsed_args.group_by
        resource_count = utils.get_client(self).resource.count(resource_type=resource_type, all_tenants=all_tenants, query=query, group_by=group_by)
        return self.dict2columns(resource_count)