import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib import utils as osc_utils
from manilaclient.common._i18n import _
class SetShareGroupSnapshot(command.Command):
    """Set share group snapshot properties."""
    _description = _('Set share group snapshot properties')

    def get_parser(self, prog_name):
        parser = super(SetShareGroupSnapshot, self).get_parser(prog_name)
        parser.add_argument('share_group_snapshot', metavar='<share-group-snapshot>', help=_('Name or ID of the snapshot to set a property for'))
        parser.add_argument('--name', metavar='<name>', default=None, help=_('Set a name to the snapshot.'))
        parser.add_argument('--description', metavar='<description>', default=None, help=_('Set a description to the snapshot.'))
        parser.add_argument('--status', metavar='<status>', choices=['available', 'error', 'creating', 'deleting', 'error_deleting'], help=_('Explicitly set the state of a share group snapshot(Admin only). Options include : available, error, creating, deleting, error_deleting.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        result = 0
        share_group_snapshot = osc_utils.find_resource(share_client.share_group_snapshots, parsed_args.share_group_snapshot)
        kwargs = {}
        if parsed_args.name is not None:
            kwargs['name'] = parsed_args.name
        if parsed_args.description is not None:
            kwargs['description'] = parsed_args.description
        if kwargs:
            try:
                share_client.share_group_snapshots.update(share_group_snapshot, **kwargs)
            except Exception as e:
                result += 1
                LOG.error(f'Failed to set name or desciption for share group snapshot with ID {share_group_snapshot.id}: {e}')
        if parsed_args.status:
            try:
                share_client.share_group_snapshots.reset_state(share_group_snapshot, parsed_args.status)
            except Exception as e:
                result += 1
                LOG.error(f'Failed to set status for share group snapshot with ID {share_group_snapshot.id}: {e}')
        if result > 0:
            raise exceptions.CommandError(_('One or more of the set operations failed'))