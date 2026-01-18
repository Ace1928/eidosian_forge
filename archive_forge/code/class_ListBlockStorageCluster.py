from cinderclient import api_versions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ListBlockStorageCluster(command.Lister):
    """List block storage clusters.

    This command requires ``--os-volume-api-version`` 3.7 or greater.
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--cluster', metavar='<name>', default=None, help=_('Filter by cluster name, without backend will list all clustered services from the same cluster.'))
        parser.add_argument('--binary', metavar='<binary>', help=_('Cluster binary.'))
        parser.add_argument('--up', action='store_true', dest='is_up', default=None, help=_('Filter by up status.'))
        parser.add_argument('--down', action='store_false', dest='is_up', help=_('Filter by down status.'))
        parser.add_argument('--disabled', action='store_true', dest='is_disabled', default=None, help=_('Filter by disabled status.'))
        parser.add_argument('--enabled', action='store_false', dest='is_disabled', help=_('Filter by enabled status.'))
        parser.add_argument('--num-hosts', metavar='<hosts>', type=int, default=None, help=_('Filter by number of hosts in the cluster.'))
        parser.add_argument('--num-down-hosts', metavar='<hosts>', type=int, default=None, help=_('Filter by number of hosts that are down.'))
        parser.add_argument('--long', action='store_true', default=False, help=_('List additional fields in output'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        if volume_client.api_version < api_versions.APIVersion('3.7'):
            msg = _("--os-volume-api-version 3.7 or greater is required to support the 'block storage cluster list' command")
            raise exceptions.CommandError(msg)
        columns = ('Name', 'Binary', 'State', 'Status')
        if parsed_args.long:
            columns += ('Num Hosts', 'Num Down Hosts', 'Last Heartbeat', 'Disabled Reason', 'Created At', 'Updated At')
        data = volume_client.clusters.list(name=parsed_args.cluster, binary=parsed_args.binary, is_up=parsed_args.is_up, disabled=parsed_args.is_disabled, num_hosts=parsed_args.num_hosts, num_down_hosts=parsed_args.num_down_hosts, detailed=parsed_args.long)
        return (columns, (utils.get_item_properties(s, columns) for s in data))