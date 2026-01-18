from cinderclient import api_versions
from osc_lib.command import command
from osc_lib import exceptions
from openstackclient.i18n import _
class BlockStorageCleanup(command.Lister):
    """Do block storage cleanup.

    This command requires ``--os-volume-api-version`` 3.24 or greater.
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--cluster', metavar='<cluster>', help=_('Name of block storage cluster in which cleanup needs to be performed (name only)'))
        parser.add_argument('--host', metavar='<host>', default=None, help=_('Host where the service resides. (name only)'))
        parser.add_argument('--binary', metavar='<binary>', default=None, help=_('Name of the service binary.'))
        service_up_parser = parser.add_mutually_exclusive_group()
        service_up_parser.add_argument('--up', dest='is_up', action='store_true', default=None, help=_('Filter by up status. If this is set, services need to be up.'))
        service_up_parser.add_argument('--down', dest='is_up', action='store_false', help=_('Filter by down status. If this is set, services need to be down.'))
        service_disabled_parser = parser.add_mutually_exclusive_group()
        service_disabled_parser.add_argument('--disabled', dest='disabled', action='store_true', default=None, help=_('Filter by disabled status.'))
        service_disabled_parser.add_argument('--enabled', dest='disabled', action='store_false', help=_('Filter by enabled status.'))
        parser.add_argument('--resource-id', metavar='<resource-id>', default=None, help=_('UUID of a resource to cleanup.'))
        parser.add_argument('--resource-type', metavar='<Volume|Snapshot>', choices=('Volume', 'Snapshot'), help=_('Type of resource to cleanup.'))
        parser.add_argument('--service-id', type=int, default=None, help=_('The service ID field from the DB, not the UUID of the service.'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        if volume_client.api_version < api_versions.APIVersion('3.24'):
            msg = _("--os-volume-api-version 3.24 or greater is required to support the 'block storage cleanup' command")
            raise exceptions.CommandError(msg)
        filters = {'cluster_name': parsed_args.cluster, 'host': parsed_args.host, 'binary': parsed_args.binary, 'is_up': parsed_args.is_up, 'disabled': parsed_args.disabled, 'resource_id': parsed_args.resource_id, 'resource_type': parsed_args.resource_type, 'service_id': parsed_args.service_id}
        filters = {k: v for k, v in filters.items() if v is not None}
        cleaning, unavailable = volume_client.workers.clean(**filters)
        return _format_cleanup_response(cleaning, unavailable)