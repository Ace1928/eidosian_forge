from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from manilaclient.common._i18n import _
from manilaclient.common import cliutils
class SetShareSnapshotInstance(command.Command):
    """Explicitly update the state of a share snapshot instance."""
    _description = _('Explicitly update the state of a share snapshot instance.')

    def get_parser(self, prog_name):
        parser = super(SetShareSnapshotInstance, self).get_parser(prog_name)
        parser.add_argument('snapshot_instance', metavar='<snapshot_instance>', help=_('ID of the share snapshot instance to update.'))
        parser.add_argument('--status', metavar='<status>', default='available', choices=['available', 'error', 'creating', 'deleting', 'error_deleting'], help=_('Indicate state to update the snapshot instance to. Default is available.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        try:
            share_client.share_snapshot_instances.reset_state(parsed_args.snapshot_instance, parsed_args.status)
        except Exception as e:
            msg = _('Failed to update share snapshot instance status: %s' % e)
            raise exceptions.CommandError(msg)