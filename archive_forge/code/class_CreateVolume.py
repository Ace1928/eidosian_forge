import argparse
import copy
import functools
import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class CreateVolume(command.ShowOne):
    _description = _('Create new volume')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('name', metavar='<name>', nargs='?', help=_('Volume name'))
        parser.add_argument('--size', metavar='<size>', type=int, help=_('Volume size in GB (required unless --snapshot, --source or --backup is specified)'))
        parser.add_argument('--type', metavar='<volume-type>', help=_('Set the type of volume'))
        source_group = parser.add_mutually_exclusive_group()
        source_group.add_argument('--image', metavar='<image>', help=_('Use <image> as source of volume (name or ID)'))
        source_group.add_argument('--snapshot', metavar='<snapshot>', help=_('Use <snapshot> as source of volume (name or ID)'))
        source_group.add_argument('--source', metavar='<volume>', help=_('Volume to clone (name or ID)'))
        source_group.add_argument('--backup', metavar='<backup>', help=_('Restore backup to a volume (name or ID) (supported by --os-volume-api-version 3.47 or later)'))
        source_group.add_argument('--source-replicated', metavar='<replicated-volume>', help=argparse.SUPPRESS)
        parser.add_argument('--description', metavar='<description>', help=_('Volume description'))
        parser.add_argument('--availability-zone', metavar='<availability-zone>', help=_('Create volume in <availability-zone>'))
        parser.add_argument('--consistency-group', metavar='consistency-group>', help=_('Consistency group where the new volume belongs to'))
        parser.add_argument('--property', metavar='<key=value>', action=parseractions.KeyValueAction, help=_('Set a property to this volume (repeat option to set multiple properties)'))
        parser.add_argument('--hint', metavar='<key=value>', action=KeyValueHintAction, help=_("Arbitrary scheduler hint key-value pairs to help creating a volume. Repeat the option to set multiple hints. 'same_host' and 'different_host' get values appended when repeated, all other keys take the last given value"))
        bootable_group = parser.add_mutually_exclusive_group()
        bootable_group.add_argument('--bootable', action='store_true', help=_('Mark volume as bootable'))
        bootable_group.add_argument('--non-bootable', action='store_true', help=_('Mark volume as non-bootable (default)'))
        readonly_group = parser.add_mutually_exclusive_group()
        readonly_group.add_argument('--read-only', action='store_true', help=_('Set volume to read-only access mode'))
        readonly_group.add_argument('--read-write', action='store_true', help=_('Set volume to read-write access mode (default)'))
        return parser

    def take_action(self, parsed_args):
        _check_size_arg(parsed_args)
        size = parsed_args.size
        volume_client = self.app.client_manager.volume
        image_client = self.app.client_manager.image
        if parsed_args.backup and (not volume_client.api_version.matches('3.47')):
            msg = _('--os-volume-api-version 3.47 or greater is required to create a volume from backup.')
            raise exceptions.CommandError(msg)
        source_volume = None
        if parsed_args.source:
            source_volume_obj = utils.find_resource(volume_client.volumes, parsed_args.source)
            source_volume = source_volume_obj.id
            size = max(size or 0, source_volume_obj.size)
        consistency_group = None
        if parsed_args.consistency_group:
            consistency_group = utils.find_resource(volume_client.consistencygroups, parsed_args.consistency_group).id
        image = None
        if parsed_args.image:
            image = image_client.find_image(parsed_args.image, ignore_missing=False).id
        snapshot = None
        if parsed_args.snapshot:
            snapshot_obj = utils.find_resource(volume_client.volume_snapshots, parsed_args.snapshot)
            snapshot = snapshot_obj.id
            size = max(size or 0, snapshot_obj.size)
        backup = None
        if parsed_args.backup:
            backup_obj = utils.find_resource(volume_client.backups, parsed_args.backup)
            backup = backup_obj.id
            size = max(size or 0, backup_obj.size)
        volume = volume_client.volumes.create(size=size, snapshot_id=snapshot, name=parsed_args.name, description=parsed_args.description, volume_type=parsed_args.type, availability_zone=parsed_args.availability_zone, metadata=parsed_args.property, imageRef=image, source_volid=source_volume, consistencygroup_id=consistency_group, scheduler_hints=parsed_args.hint, backup_id=backup)
        if parsed_args.bootable or parsed_args.non_bootable:
            try:
                if utils.wait_for_status(volume_client.volumes.get, volume.id, success_status=['available'], error_status=['error'], sleep_time=1):
                    volume_client.volumes.set_bootable(volume.id, parsed_args.bootable)
                else:
                    msg = _('Volume status is not available for setting boot state')
                    raise exceptions.CommandError(msg)
            except Exception as e:
                LOG.error(_('Failed to set volume bootable property: %s'), e)
        if parsed_args.read_only or parsed_args.read_write:
            try:
                if utils.wait_for_status(volume_client.volumes.get, volume.id, success_status=['available'], error_status=['error'], sleep_time=1):
                    volume_client.volumes.update_readonly_flag(volume.id, parsed_args.read_only)
                else:
                    msg = _('Volume status is not available for setting itread only.')
                    raise exceptions.CommandError(msg)
            except Exception as e:
                LOG.error(_('Failed to set volume read-only access mode flag: %s'), e)
        volume._info.update({'properties': format_columns.DictColumn(volume._info.pop('metadata')), 'type': volume._info.pop('volume_type')})
        volume._info.pop('links', None)
        return zip(*sorted(volume._info.items()))