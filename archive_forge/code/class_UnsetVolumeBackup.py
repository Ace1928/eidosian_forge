import copy
import functools
import logging
from cinderclient import api_versions
from cliff import columns as cliff_columns
from openstack import utils as sdk_utils
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
class UnsetVolumeBackup(command.Command):
    """Unset volume backup properties.

    This command requires ``--os-volume-api-version`` 3.43 or greater.
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('backup', metavar='<backup>', help=_('Backup to modify (name or ID)'))
        parser.add_argument('--property', metavar='<key>', action='append', dest='properties', help=_('Property to remove from this backup (repeat option to unset multiple values) '))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        if volume_client.api_version < api_versions.APIVersion('3.43'):
            msg = _('--os-volume-api-version 3.43 or greater is required to support the --property option')
            raise exceptions.CommandError(msg)
        backup = utils.find_resource(volume_client.backups, parsed_args.backup)
        metadata = copy.deepcopy(backup.metadata)
        for key in parsed_args.properties:
            if key not in metadata:
                LOG.warning("'%s' is not a valid property for backup '%s'", key, parsed_args.backup)
                continue
            del metadata[key]
        kwargs = {'metadata': metadata}
        volume_client.backups.update(backup.id, **kwargs)