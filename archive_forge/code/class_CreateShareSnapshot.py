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
class CreateShareSnapshot(command.ShowOne):
    """Create a share snapshot."""
    _description = _('Create a snapshot of the given share')

    def get_parser(self, prog_name):
        parser = super(CreateShareSnapshot, self).get_parser(prog_name)
        parser.add_argument('share', metavar='<share>', help=_('Name or ID of the share to create snapshot of'))
        parser.add_argument('--force', action='store_true', default=False, help=_("Optional flag to indicate whether to snapshot a share even if it's busy. (Default=False)"))
        parser.add_argument('--name', metavar='<name>', default=None, help=_('Add a name to the snapshot (Optional).'))
        parser.add_argument('--description', metavar='<description>', default=None, help=_('Add a description to the snapshot (Optional).'))
        parser.add_argument('--wait', action='store_true', default=False, help=_('Wait for share snapshot creation'))
        parser.add_argument('--property', metavar='<key=value>', default={}, action=parseractions.KeyValueAction, help=_('Set a property to this snapshot (repeat option to set multiple properties).Available only for microversion >= 2.73'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share = utils.find_resource(share_client.shares, parsed_args.share)
        if share_client.api_version >= api_versions.APIVersion('2.73'):
            property = parsed_args.property or {}
        elif parsed_args.property:
            raise exceptions.CommandError('Setting metadtaa is only available with manila API version >= 2.73')
        share_snapshot = share_client.share_snapshots.create(share=share, force=parsed_args.force, name=parsed_args.name or None, description=parsed_args.description or None, metadata=property)
        if parsed_args.wait:
            if not utils.wait_for_status(status_f=share_client.share_snapshots.get, res_id=share_snapshot.id, success_status=['available']):
                LOG.error(_('ERROR: Share snapshot is in error state.'))
            share_snapshot = utils.find_resource(share_client.share_snapshots, share_snapshot.id)
        share_snapshot._info.pop('links', None)
        return self.dict2columns(share_snapshot._info)