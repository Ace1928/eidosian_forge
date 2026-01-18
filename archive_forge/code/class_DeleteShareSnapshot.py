import logging
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common import cliutils
from manilaclient.osc import utils as oscutils
class DeleteShareSnapshot(command.Command):
    """Delete one or more share snapshots"""
    _description = _('Delete one or more share snapshots')

    def get_parser(self, prog_name):
        parser = super(DeleteShareSnapshot, self).get_parser(prog_name)
        parser.add_argument('snapshot', metavar='<snapshot>', nargs='+', help=_('Name or ID of the snapshot(s) to delete'))
        parser.add_argument('--force', action='store_true', default=False, help=_('Delete the snapshot(s) ignoring the current state.'))
        parser.add_argument('--wait', action='store_true', default=False, help=_('Wait for share snapshot deletion'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        result = 0
        for snapshot in parsed_args.snapshot:
            try:
                snapshot_obj = utils.find_resource(share_client.share_snapshots, snapshot)
                if parsed_args.force:
                    share_client.share_snapshots.force_delete(snapshot_obj)
                else:
                    share_client.share_snapshots.delete(snapshot_obj)
                if parsed_args.wait:
                    if not utils.wait_for_delete(manager=share_client.share_snapshots, res_id=snapshot_obj.id):
                        result += 1
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete snapshot with name or ID '%(snapshot)s': %(e)s"), {'snapshot': snapshot, 'e': e})
        if result > 0:
            total = len(parsed_args.snapshot)
            msg = _('%(result)s of %(total)s snapshots failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)