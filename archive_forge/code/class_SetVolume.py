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
class SetVolume(command.Command):
    _description = _('Set volume properties')

    def get_parser(self, prog_name):
        parser = super(SetVolume, self).get_parser(prog_name)
        parser.add_argument('volume', metavar='<volume>', help=_('Volume to modify (name or ID)'))
        parser.add_argument('--name', metavar='<name>', help=_('New volume name'))
        parser.add_argument('--size', metavar='<size>', type=int, help=_('Extend volume size in GB'))
        parser.add_argument('--description', metavar='<description>', help=_('New volume description'))
        parser.add_argument('--no-property', dest='no_property', action='store_true', help=_('Remove all properties from <volume> (specify both --no-property and --property to remove the current properties before setting new properties.)'))
        parser.add_argument('--property', metavar='<key=value>', action=parseractions.KeyValueAction, help=_('Set a property on this volume (repeat option to set multiple properties)'))
        parser.add_argument('--image-property', metavar='<key=value>', action=parseractions.KeyValueAction, help=_('Set an image property on this volume (repeat option to set multiple image properties)'))
        parser.add_argument('--state', metavar='<state>', choices=['available', 'error', 'creating', 'deleting', 'in-use', 'attaching', 'detaching', 'error_deleting', 'maintenance'], help=_('New volume state ("available", "error", "creating", "deleting", "in-use", "attaching", "detaching", "error_deleting" or "maintenance") (admin only) (This option simply changes the state of the volume in the database with no regard to actual status, exercise caution when using)'))
        attached_group = parser.add_mutually_exclusive_group()
        attached_group.add_argument('--attached', action='store_true', help=_('Set volume attachment status to "attached" (admin only) (This option simply changes the state of the volume in the database with no regard to actual status, exercise caution when using)'))
        attached_group.add_argument('--detached', action='store_true', help=_('Set volume attachment status to "detached" (admin only) (This option simply changes the state of the volume in the database with no regard to actual status, exercise caution when using)'))
        parser.add_argument('--type', metavar='<volume-type>', help=_('New volume type (name or ID)'))
        parser.add_argument('--retype-policy', metavar='<retype-policy>', choices=['never', 'on-demand'], help=_('Migration policy while re-typing volume ("never" or "on-demand", default is "never" ) (available only when --type option is specified)'))
        bootable_group = parser.add_mutually_exclusive_group()
        bootable_group.add_argument('--bootable', action='store_true', help=_('Mark volume as bootable'))
        bootable_group.add_argument('--non-bootable', action='store_true', help=_('Mark volume as non-bootable'))
        readonly_group = parser.add_mutually_exclusive_group()
        readonly_group.add_argument('--read-only', action='store_true', help=_('Set volume to read-only access mode'))
        readonly_group.add_argument('--read-write', action='store_true', help=_('Set volume to read-write access mode'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        volume = utils.find_resource(volume_client.volumes, parsed_args.volume)
        result = 0
        if parsed_args.size:
            try:
                if parsed_args.size <= volume.size:
                    msg = _('New size must be greater than %s GB') % volume.size
                    raise exceptions.CommandError(msg)
                if volume.status != 'available' and (not volume_client.api_version.matches('3.42')):
                    msg = _('Volume is in %s state, it must be available before size can be extended') % volume.status
                    raise exceptions.CommandError(msg)
                volume_client.volumes.extend(volume.id, parsed_args.size)
            except Exception as e:
                LOG.error(_('Failed to set volume size: %s'), e)
                result += 1
        if parsed_args.no_property:
            try:
                volume_client.volumes.delete_metadata(volume.id, volume.metadata.keys())
            except Exception as e:
                LOG.error(_('Failed to clean volume properties: %s'), e)
                result += 1
        if parsed_args.property:
            try:
                volume_client.volumes.set_metadata(volume.id, parsed_args.property)
            except Exception as e:
                LOG.error(_('Failed to set volume property: %s'), e)
                result += 1
        if parsed_args.image_property:
            try:
                volume_client.volumes.set_image_metadata(volume.id, parsed_args.image_property)
            except Exception as e:
                LOG.error(_('Failed to set image property: %s'), e)
                result += 1
        if parsed_args.state:
            try:
                volume_client.volumes.reset_state(volume.id, parsed_args.state)
            except Exception as e:
                LOG.error(_('Failed to set volume state: %s'), e)
                result += 1
        if parsed_args.attached:
            try:
                volume_client.volumes.reset_state(volume.id, state=None, attach_status='attached')
            except Exception as e:
                LOG.error(_('Failed to set volume attach-status: %s'), e)
                result += 1
        if parsed_args.detached:
            try:
                volume_client.volumes.reset_state(volume.id, state=None, attach_status='detached')
            except Exception as e:
                LOG.error(_('Failed to set volume attach-status: %s'), e)
                result += 1
        if parsed_args.bootable or parsed_args.non_bootable:
            try:
                volume_client.volumes.set_bootable(volume.id, parsed_args.bootable)
            except Exception as e:
                LOG.error(_('Failed to set volume bootable property: %s'), e)
                result += 1
        if parsed_args.read_only or parsed_args.read_write:
            try:
                volume_client.volumes.update_readonly_flag(volume.id, parsed_args.read_only)
            except Exception as e:
                LOG.error(_('Failed to set volume read-only access mode flag: %s'), e)
                result += 1
        if parsed_args.type:
            migration_policy = 'never'
            if parsed_args.retype_policy:
                migration_policy = parsed_args.retype_policy
            try:
                volume_type = utils.find_resource(volume_client.volume_types, parsed_args.type)
                volume_client.volumes.retype(volume.id, volume_type.id, migration_policy)
            except Exception as e:
                LOG.error(_('Failed to set volume type: %s'), e)
                result += 1
        elif parsed_args.retype_policy:
            LOG.warning(_("'--retype-policy' option will not work without '--type' option"))
        kwargs = {}
        if parsed_args.name:
            kwargs['display_name'] = parsed_args.name
        if parsed_args.description:
            kwargs['display_description'] = parsed_args.description
        if kwargs:
            try:
                volume_client.volumes.update(volume.id, **kwargs)
            except Exception as e:
                LOG.error(_('Failed to update volume display name or display description: %s'), e)
                result += 1
        if result > 0:
            raise exceptions.CommandError(_('One or more of the set operations failed'))