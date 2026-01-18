import os
from magnumclient.common import utils as magnum_utils
from magnumclient import exceptions
from magnumclient.i18n import _
from osc_lib.command import command
from osc_lib import utils
class ShowCluster(command.ShowOne):
    _description = _('Show a Cluster')

    def get_parser(self, prog_name):
        parser = super(ShowCluster, self).get_parser(prog_name)
        parser.add_argument('cluster', metavar='<cluster>', help=_('ID or name of the cluster to show.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        columns = CLUSTER_ATTRIBUTES
        mag_client = self.app.client_manager.container_infra
        cluster = mag_client.clusters.get(parsed_args.cluster)
        return (columns, utils.get_item_properties(cluster, columns))