import os
from magnumclient.i18n import _
from osc_lib.command import command
class RotateCa(command.Command):
    _description = _('Rotate the CA certificate for cluster to revoke access.')

    def get_parser(self, prog_name):
        parser = super(RotateCa, self).get_parser(prog_name)
        parser.add_argument('cluster', metavar='<cluster>', help='ID or name of the cluster')
        return parser

    def take_action(self, parsed_args):
        mag_client = self.app.client_manager.container_infra
        cluster = mag_client.clusters.get(parsed_args.cluster)
        opts = {'cluster_uuid': cluster.uuid}
        mag_client.certificates.rotate_ca(**opts)
        print('Request to rotate the CA certificate for cluster %s has been accepted.' % cluster.uuid)