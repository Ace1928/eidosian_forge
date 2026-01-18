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
class SetShareSnapshot(command.Command):
    """Set share snapshot properties."""
    _description = _('Set share snapshot properties')

    def get_parser(self, prog_name):
        parser = super(SetShareSnapshot, self).get_parser(prog_name)
        parser.add_argument('snapshot', metavar='<snapshot>', help=_('Name or ID of the snapshot to set a property for'))
        parser.add_argument('--name', metavar='<name>', default=None, help=_('Set a name to the snapshot.'))
        parser.add_argument('--description', metavar='<description>', default=None, help=_('Set a description to the snapshot.'))
        parser.add_argument('--status', metavar='<status>', choices=['available', 'error', 'creating', 'deleting', 'manage_starting', 'manage_error', 'unmanage_starting', 'unmanage_error', 'error_deleting'], help=_('Assign a status to the snapshot (Admin only). Options include : available, error, creating, deleting, manage_starting, manage_error, unmanage_starting, unmanage_error, error_deleting.'))
        parser.add_argument('--property', metavar='<key=value>', default={}, action=parseractions.KeyValueAction, help=_('Set a property to this snapshot (repeat option to set multiple properties)'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        result = 0
        share_snapshot = utils.find_resource(share_client.share_snapshots, parsed_args.snapshot)
        kwargs = {}
        if parsed_args.name is not None:
            kwargs['display_name'] = parsed_args.name
        if parsed_args.description is not None:
            kwargs['display_description'] = parsed_args.description
        try:
            share_client.share_snapshots.update(share_snapshot, **kwargs)
        except Exception as e:
            result += 1
            LOG.error(_("Failed to set share snapshot properties '%(properties)s': %(exception)s"), {'properties': kwargs, 'exception': e})
        if parsed_args.status:
            try:
                share_client.share_snapshots.reset_state(share_snapshot, parsed_args.status)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to update snapshot status to '%(status)s': %(e)s"), {'status': parsed_args.status, 'e': e})
        if parsed_args.property:
            try:
                share_snapshot.set_metadata(parsed_args.property)
            except Exception as e:
                LOG.error(_("Failed to set share snapshot properties '%(properties)s': %(exception)s"), {'properties': parsed_args.property, 'exception': e})
                result += 1
        if result > 0:
            raise exceptions.CommandError(_('One or more of the set operations failed'))