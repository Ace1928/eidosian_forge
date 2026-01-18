from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
from zunclient.common import template_utils
from zunclient.common import utils as zun_utils
from zunclient.i18n import _
class ListCapsule(command.Lister):
    """List available capsules"""
    log = logging.getLogger(__name__ + '.ListCapsule')

    def get_parser(self, prog_name):
        parser = super(ListCapsule, self).get_parser(prog_name)
        parser.add_argument('--all-projects', action='store_true', default=False, help='List capsules in all projects')
        parser.add_argument('--marker', metavar='<marker>', help='The last capsule UUID of the previous page; displays list of capsules after "marker".')
        parser.add_argument('--limit', metavar='<limit>', type=int, help='Maximum number of capsules to return')
        parser.add_argument('--sort-key', metavar='<sort-key>', help='Column to sort results by')
        parser.add_argument('--sort-dir', metavar='<sort-dir>', choices=['desc', 'asc'], help='Direction to sort. "asc" or "desc".')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        opts = {}
        opts['all_projects'] = parsed_args.all_projects
        opts['marker'] = parsed_args.marker
        opts['limit'] = parsed_args.limit
        opts['sort_key'] = parsed_args.sort_key
        opts['sort_dir'] = parsed_args.sort_dir
        opts = zun_utils.remove_null_parms(**opts)
        capsules = client.capsules.list(**opts)
        for capsule in capsules:
            zun_utils.format_container_addresses(capsule)
        columns = ('uuid', 'name', 'status', 'addresses')
        return (columns, (utils.get_item_properties(capsule, columns) for capsule in capsules))