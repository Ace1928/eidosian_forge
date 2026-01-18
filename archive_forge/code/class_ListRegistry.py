from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
from zunclient.common import utils as zun_utils
from zunclient import exceptions as exc
from zunclient.i18n import _
class ListRegistry(command.Lister):
    """List available registries"""
    log = logging.getLogger(__name__ + '.ListRegistrys')

    def get_parser(self, prog_name):
        parser = super(ListRegistry, self).get_parser(prog_name)
        parser.add_argument('--all-projects', action='store_true', default=False, help='List registries in all projects')
        parser.add_argument('--marker', metavar='<marker>', help='The last registry UUID of the previous page; displays list of registries after "marker".')
        parser.add_argument('--limit', metavar='<limit>', type=int, help='Maximum number of registries to return')
        parser.add_argument('--sort-key', metavar='<sort-key>', help='Column to sort results by')
        parser.add_argument('--sort-dir', metavar='<sort-dir>', choices=['desc', 'asc'], help='Direction to sort. "asc" or "desc".')
        parser.add_argument('--name', metavar='<name>', help='List registries according to their name.')
        parser.add_argument('--domain', metavar='<domain>', help='List registries according to their domain.')
        parser.add_argument('--project-id', metavar='<project-id>', help='List registries according to their project_id')
        parser.add_argument('--user-id', metavar='<user-id>', help='List registries according to their user_id')
        parser.add_argument('--username', metavar='<username>', help='List registries according to their username')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        opts = {}
        opts['all_projects'] = parsed_args.all_projects
        opts['marker'] = parsed_args.marker
        opts['limit'] = parsed_args.limit
        opts['sort_key'] = parsed_args.sort_key
        opts['sort_dir'] = parsed_args.sort_dir
        opts['domain'] = parsed_args.domain
        opts['name'] = parsed_args.name
        opts['project_id'] = parsed_args.project_id
        opts['user_id'] = parsed_args.user_id
        opts['username'] = parsed_args.username
        opts = zun_utils.remove_null_parms(**opts)
        registries = client.registries.list(**opts)
        columns = ('uuid', 'name', 'domain', 'username', 'password')
        return (columns, (utils.get_item_properties(registry, columns) for registry in registries))