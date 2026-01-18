import argparse
from cinderclient import api_versions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class BlockStorageManageVolumes(command.Lister):
    """List manageable volumes.

    Supported by --os-volume-api-version 3.8 or greater.
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        host_group = parser.add_mutually_exclusive_group()
        host_group.add_argument('host', metavar='<host>', nargs='?', help=_('Cinder host on which to list manageable volumes. Takes the form: host@backend-name#pool'))
        host_group.add_argument('--cluster', metavar='<cluster>', help=_('Cinder cluster on which to list manageable volumes. Takes the form: cluster@backend-name#pool. (supported by --os-volume-api-version 3.17 or later)'))
        parser.add_argument('--long', action='store_true', default=False, help=_('List additional fields in output'))
        parser.add_argument('--detailed', metavar='<detailed>', default=None, help=argparse.SUPPRESS)
        parser.add_argument('--marker', metavar='<marker>', default=None, help=_('Begin returning volumes that appear later in the volume list than that represented by this reference. This reference should be json like. Default=None.'))
        parser.add_argument('--limit', metavar='<limit>', default=None, help=_('Maximum number of volumes to return. Default=None.'))
        parser.add_argument('--offset', metavar='<offset>', default=None, help=_('Number of volumes to skip after marker. Default=None.'))
        parser.add_argument('--sort', metavar='<key>[:<direction>]', default=None, help=_('Comma-separated list of sort keys and directions in the form of <key>[:<asc|desc>]. Valid keys: %s. Default=None.') % ', '.join(SORT_MANAGEABLE_KEY_VALUES))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        if parsed_args.host is None and parsed_args.cluster is None:
            msg = _("Either <host> or '--cluster <cluster>' needs to be provided to run the 'block storage volume manageable list' command")
            raise exceptions.CommandError(msg)
        if volume_client.api_version < api_versions.APIVersion('3.8'):
            msg = _("--os-volume-api-version 3.8 or greater is required to support the 'block storage volume manageable list' command")
            raise exceptions.CommandError(msg)
        if parsed_args.cluster:
            if volume_client.api_version < api_versions.APIVersion('3.17'):
                msg = _("--os-volume-api-version 3.17 or greater is required to support the '--cluster' option")
                raise exceptions.CommandError(msg)
        detailed = parsed_args.long
        if parsed_args.detailed is not None:
            detailed = parsed_args.detailed.lower().strip() in {'1', 't', 'true', 'on', 'y', 'yes'}
            if detailed:
                msg = _('The --detailed option has been deprecated. Use --long instead.')
                self.log.warning(msg)
            else:
                msg = _('The --detailed option has been deprecated. Unset it.')
                self.log.warning(msg)
        columns = ['reference', 'size', 'safe_to_manage']
        if detailed:
            columns.extend(['reason_not_safe', 'cinder_id', 'extra_info'])
        data = volume_client.volumes.list_manageable(host=parsed_args.host, detailed=detailed, marker=parsed_args.marker, limit=parsed_args.limit, offset=parsed_args.offset, sort=parsed_args.sort, cluster=parsed_args.cluster)
        return (columns, (utils.get_item_properties(s, columns) for s in data))