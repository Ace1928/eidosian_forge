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
class ShrinkDatabaseCluster(command.Command):
    _description = _('Drops instances from a cluster.')

    def get_parser(self, prog_name):
        parser = super(ShrinkDatabaseCluster, self).get_parser(prog_name)
        parser.add_argument('cluster', metavar='<cluster>', help=_('ID or name of the cluster.'))
        parser.add_argument('instances', metavar='<instance>', nargs='+', default=[], help=_('Drop instance(s) from the cluster. Specify multiple ids to drop multiple instances.'))
        return parser

    def take_action(self, parsed_args):
        database_client_manager = self.app.client_manager.database
        db_clusters = database_client_manager.clusters
        cluster = utils.find_resource(db_clusters, parsed_args.cluster)
        db_instances = database_client_manager.instances
        instances = [{'id': utils.find_resource(db_instances, instance).id} for instance in parsed_args.instances]
        db_clusters.shrink(cluster, instances)