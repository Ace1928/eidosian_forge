from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from troveclient.i18n import _
from troveclient.v1.shell import _parse_extended_properties
from troveclient.v1.shell import _parse_instance_options
from troveclient.v1.shell import EXT_PROPS_HELP
from troveclient.v1.shell import EXT_PROPS_METAVAR
from troveclient.v1.shell import INSTANCE_HELP
from troveclient.v1.shell import INSTANCE_METAVAR
class ListDatabaseClusters(command.Lister):
    _description = _('List database clusters')
    columns = ['ID', 'Name', 'Datastore', 'Datastore Version', 'Task Name']

    def get_parser(self, prog_name):
        parser = super(ListDatabaseClusters, self).get_parser(prog_name)
        parser.add_argument('--limit', dest='limit', metavar='<limit>', type=int, default=None, help=_('Limit the number of results displayed.'))
        parser.add_argument('--marker', dest='marker', metavar='<ID>', type=str, default=None, help=_('Begin displaying the results for IDs greater than the specified marker. When used with ``--limit``, set  this to the last ID displayed in the previous run.'))
        return parser

    def take_action(self, parsed_args):
        database_clusters = self.app.client_manager.database.clusters
        clusters = database_clusters.list(limit=parsed_args.limit, marker=parsed_args.marker)
        for cluster in clusters:
            setattr(cluster, 'datastore_version', cluster.datastore['version'])
            setattr(cluster, 'datastore', cluster.datastore['type'])
            setattr(cluster, 'task_name', cluster.task['name'])
        clusters = [utils.get_item_properties(c, self.columns) for c in clusters]
        return (self.columns, clusters)