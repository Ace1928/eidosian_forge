from magnumclient.common import cliutils as utils
from magnumclient.common import utils as magnum_utils
from magnumclient.i18n import _
from osc_lib.command import command
class ListQuotas(command.Command):
    _description = _('Print a list of available quotas.')

    def get_parser(self, prog_name):
        parser = super(ListQuotas, self).get_parser(prog_name)
        parser.add_argument('--marker', metavar='<marker>', default=None, help=_('The last quota UUID of the previous page; displays list of quotas after "marker".'))
        parser.add_argument('--limit', metavar='<limit>', type=int, help='Maximum number of quotas to return.')
        parser.add_argument('--sort-key', metavar='<sort-key>', help='Column to sort results by.')
        parser.add_argument('--sort-dir', metavar='<sort-dir>', choices=['desc', 'asc'], help='Direction to sort. "asc" or "desc".')
        parser.add_argument('--all-tenants', action='store_true', default=False, help='Flag to indicate list all tenant quotas.')
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        mag_client = self.app.client_manager.container_infra
        quotas = mag_client.quotas.list(marker=parsed_args.marker, limit=parsed_args.limit, sort_key=parsed_args.sort_key, sort_dir=parsed_args.sort_dir, all_tenants=parsed_args.all_tenants)
        columns = ['project_id', 'resource', 'hard_limit']
        utils.print_list(quotas, columns, {'versions': magnum_utils.print_list_field('versions')}, sortby_index=None)