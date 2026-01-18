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
class CreateVolumeSnapshot(command.ShowOne):
    _description = _('Create new volume snapshot')

    def get_parser(self, prog_name):
        parser = super(CreateVolumeSnapshot, self).get_parser(prog_name)
        parser.add_argument('snapshot_name', metavar='<snapshot-name>', help=_('Name of the new snapshot'))
        parser.add_argument('--volume', metavar='<volume>', help=_('Volume to snapshot (name or ID) (default is <snapshot-name>)'))
        parser.add_argument('--description', metavar='<description>', help=_('Description of the snapshot'))
        parser.add_argument('--force', action='store_true', default=False, help=_('Create a snapshot attached to an instance. Default is False'))
        parser.add_argument('--property', metavar='<key=value>', action=parseractions.KeyValueAction, help=_('Set a property to this snapshot (repeat option to set multiple properties)'))
        parser.add_argument('--remote-source', metavar='<key=value>', action=parseractions.KeyValueAction, help=_("The attribute(s) of the existing remote volume snapshot (admin required) (repeat option to specify multiple attributes) e.g.: '--remote-source source-name=test_name --remote-source source-id=test_id'"))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        volume = parsed_args.volume
        if not parsed_args.volume:
            volume = parsed_args.snapshot_name
        volume_id = utils.find_resource(volume_client.volumes, volume).id
        if parsed_args.remote_source:
            if parsed_args.force:
                msg = _("'--force' option will not work when you create new volume snapshot from an existing remote volume snapshot")
                LOG.warning(msg)
            snapshot = volume_client.volume_snapshots.manage(volume_id=volume_id, ref=parsed_args.remote_source, name=parsed_args.snapshot_name, description=parsed_args.description, metadata=parsed_args.property)
        else:
            snapshot = volume_client.volume_snapshots.create(volume_id, force=parsed_args.force, name=parsed_args.snapshot_name, description=parsed_args.description, metadata=parsed_args.property)
        snapshot._info.update({'properties': format_columns.DictColumn(snapshot._info.pop('metadata'))})
        return zip(*sorted(snapshot._info.items()))