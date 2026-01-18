import os
from magnumclient.common import utils as magnum_utils
from magnumclient import exceptions
from magnumclient.i18n import _
from osc_lib.command import command
from osc_lib import utils
class ListCluster(command.Lister):
    _description = _('List clusters')

    def get_parser(self, prog_name):
        parser = super(ListCluster, self).get_parser(prog_name)
        parser.add_argument('--limit', metavar='<limit>', type=int, help=_('Maximum number of clusters to return'))
        parser.add_argument('--sort-key', metavar='<sort-key>', help=_('Column to sort results by'))
        parser.add_argument('--sort-dir', metavar='<sort-dir>', choices=['desc', 'asc'], help=_('Direction to sort. "asc" or "desc".'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        mag_client = self.app.client_manager.container_infra
        columns = ['uuid', 'name', 'keypair', 'node_count', 'master_count', 'status', 'health_status']
        clusters = mag_client.clusters.list(limit=parsed_args.limit, sort_key=parsed_args.sort_key, sort_dir=parsed_args.sort_dir)
        return (columns, (utils.get_item_properties(c, columns) for c in clusters))