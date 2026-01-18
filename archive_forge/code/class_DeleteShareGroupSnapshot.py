import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib import utils as osc_utils
from manilaclient.common._i18n import _
class DeleteShareGroupSnapshot(command.Command):
    """Delete one or more share group snapshots."""
    _description = _('Delete one or more share group snapshot')

    def get_parser(self, prog_name):
        parser = super(DeleteShareGroupSnapshot, self).get_parser(prog_name)
        parser.add_argument('share_group_snapshot', metavar='<share-group-snapshot>', nargs='+', help=_('Name or ID of the group snapshot(s) to delete'))
        parser.add_argument('--force', action='store_true', default=False, help=_('Attempt to force delete the share group snapshot(s) (Default=False) (Admin only).'))
        parser.add_argument('--wait', action='store_true', default=False, help=_('Wait for share group snapshot deletion'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        result = 0
        for share_group_snapshot in parsed_args.share_group_snapshot:
            try:
                share_group_snapshot_obj = osc_utils.find_resource(share_client.share_group_snapshots, share_group_snapshot)
                share_client.share_group_snapshots.delete(share_group_snapshot_obj, force=parsed_args.force)
                if parsed_args.wait:
                    if not osc_utils.wait_for_delete(manager=share_client.share_group_snapshots, res_id=share_group_snapshot_obj.id):
                        result += 1
            except Exception as e:
                result += 1
                LOG.error(f'Failed to delete a share group snapshot with name or ID {share_group_snapshot}: {e}')
        if result > 0:
            total = len(parsed_args.share_group_snapshot)
            msg = f'{result} of {total} share group snapshots failed to delete.'
            raise exceptions.CommandError(msg)