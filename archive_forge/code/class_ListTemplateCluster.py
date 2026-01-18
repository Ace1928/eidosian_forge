from magnumclient.common import utils as magnum_utils
from magnumclient.exceptions import InvalidAttribute
from magnumclient.i18n import _
from osc_lib.command import command
from osc_lib import utils as osc_utils
from oslo_log import log as logging
class ListTemplateCluster(command.Lister):
    """List Cluster Templates."""
    _description = _('List Cluster Templates.')
    log = logging.getLogger(__name__ + '.ListTemplateCluster')

    def get_parser(self, prog_name):
        parser = super(ListTemplateCluster, self).get_parser(prog_name)
        parser.add_argument('--limit', metavar='<limit>', type=int, help=_('Maximum number of cluster templates to return'))
        parser.add_argument('--sort-key', metavar='<sort-key>', help=_('Column to sort results by'))
        parser.add_argument('--sort-dir', metavar='<sort-dir>', choices=['desc', 'asc'], help=_('Direction to sort. "asc" or "desc".'))
        parser.add_argument('--fields', default=None, metavar='<fields>', help=_('Comma-separated list of fields to display. Available fields: uuid, name, coe, image_id, public, link, apiserver_port, server_type, tls_disabled, registry_enabled'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        mag_client = self.app.client_manager.container_infra
        columns = ['uuid', 'name', 'tags']
        if parsed_args.fields:
            columns += parsed_args.fields.split(',')
        cts = mag_client.cluster_templates.list(limit=parsed_args.limit, sort_key=parsed_args.sort_key, sort_dir=parsed_args.sort_dir)
        return (columns, (osc_utils.get_item_properties(ct, columns) for ct in cts))