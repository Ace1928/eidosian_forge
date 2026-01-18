import os
from magnumclient.common import utils as magnum_utils
from magnumclient import exceptions
from magnumclient.i18n import _
from osc_lib.command import command
from osc_lib import utils
class ResizeCluster(command.Command):
    _description = _('Resize a Cluster')

    def get_parser(self, prog_name):
        parser = super(ResizeCluster, self).get_parser(prog_name)
        parser.add_argument('cluster', metavar='<cluster>', help=_('The name or UUID of cluster to update'))
        parser.add_argument('node_count', type=int, help=_('Desired node count of the cluser.'))
        parser.add_argument('--nodes-to-remove', metavar='<Server UUID>', action='append', help=_('Server ID of the nodes to be removed. Repeat to add more server ID'))
        parser.add_argument('--nodegroup', metavar='<nodegroup>', help=_('The name or UUID of the nodegroup of current cluster.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        mag_client = self.app.client_manager.container_infra
        cluster = mag_client.clusters.get(parsed_args.cluster)
        mag_client.clusters.resize(cluster.uuid, parsed_args.node_count, parsed_args.nodes_to_remove, parsed_args.nodegroup)
        print('Request to resize cluster %s has been accepted.' % parsed_args.cluster)