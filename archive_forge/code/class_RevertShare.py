import logging
from openstackclient.identity import common as identity_common
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import exceptions as apiclient_exceptions
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.common import cliutils
from manilaclient.osc import utils
class RevertShare(command.Command):
    """Revert a share to snapshot."""
    _description = _('Revert a share to the specified snapshot.')

    def get_parser(self, prog_name):
        parser = super(RevertShare, self).get_parser(prog_name)
        parser.add_argument('snapshot', metavar='<snapshot>', help=_('Name or ID of the snapshot to restore. The snapshot must be the most recent one known to manila.'))
        parser.add_argument('--wait', action='store_true', default=False, help=_('Wait for share revert'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        snapshot = apiutils.find_resource(share_client.share_snapshots, parsed_args.snapshot)
        share = apiutils.find_resource(share_client.shares, snapshot.share_id)
        try:
            share.revert_to_snapshot(snapshot)
        except Exception as e:
            raise exceptions.CommandError(_('Failed to revert share to snapshot: %s' % e))
        if parsed_args.wait:
            if not oscutils.wait_for_status(status_f=share_client.shares.get, res_id=share.id, success_status=['available']):
                raise exceptions.CommandError(_('Share not available after revert attempt.'))