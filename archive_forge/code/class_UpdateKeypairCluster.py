import sys
from osc_lib.command import command
from osc_lib import utils as osc_utils
from oslo_log import log as logging
from saharaclient.osc import utils
from saharaclient.osc.v1 import clusters as c_v1
class UpdateKeypairCluster(command.ShowOne):
    """Reflects an updated keypair on the cluster"""
    log = logging.getLogger(__name__ + '.UpdateKeypairCluster')

    def get_parser(self, prog_name):
        parser = super(UpdateKeypairCluster, self).get_parser(prog_name)
        parser.add_argument('cluster', metavar='<cluster>', help='Name or ID of the cluster')
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        client = self.app.client_manager.data_processing
        cluster_id = utils.get_resource_id(client.clusters, parsed_args.cluster)
        client.clusters.update_keypair(cluster_id)
        sys.stdout.write('Cluster "{cluster}" keypair has been updated.\n'.format(cluster=parsed_args.cluster))
        return ({}, {})