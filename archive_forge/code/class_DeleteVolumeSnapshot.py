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
class DeleteVolumeSnapshot(command.Command):
    _description = _('Delete volume snapshot(s)')

    def get_parser(self, prog_name):
        parser = super(DeleteVolumeSnapshot, self).get_parser(prog_name)
        parser.add_argument('snapshots', metavar='<snapshot>', nargs='+', help=_('Snapshot(s) to delete (name or ID)'))
        parser.add_argument('--force', action='store_true', help=_('Attempt forced removal of snapshot(s), regardless of state (defaults to False)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        result = 0
        for i in parsed_args.snapshots:
            try:
                snapshot_id = utils.find_resource(volume_client.volume_snapshots, i).id
                volume_client.volume_snapshots.delete(snapshot_id, parsed_args.force)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete snapshot with name or ID '%(snapshot)s': %(e)s") % {'snapshot': i, 'e': e})
        if result > 0:
            total = len(parsed_args.snapshots)
            msg = _('%(result)s of %(total)s snapshots failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)