from cinderclient import api_versions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ShowBlockStorageCluster(command.ShowOne):
    """Show detailed information for a block storage cluster.

    This command requires ``--os-volume-api-version`` 3.7 or greater.
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('cluster', metavar='<cluster>', help=_('Name of block storage cluster.'))
        parser.add_argument('--binary', metavar='<binary>', help=_('Service binary.'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        if volume_client.api_version < api_versions.APIVersion('3.7'):
            msg = _("--os-volume-api-version 3.7 or greater is required to support the 'block storage cluster show' command")
            raise exceptions.CommandError(msg)
        cluster = volume_client.clusters.show(parsed_args.cluster, binary=parsed_args.binary)
        return _format_cluster(cluster, detailed=True)