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
class SetShare(command.Command):
    """Set share properties."""
    _description = _('Set share properties')

    def get_parser(self, prog_name):
        parser = super(SetShare, self).get_parser(prog_name)
        parser.add_argument('share', metavar='<share>', help=_('Share to modify (name or ID)'))
        parser.add_argument('--property', metavar='<key=value>', default={}, action=parseractions.KeyValueAction, help=_('Set a property to this share (repeat option to set multiple properties)'))
        parser.add_argument('--name', metavar='<name>', default=None, help=_('New share name. (Default=None)'))
        parser.add_argument('--description', metavar='<description>', default=None, help=_('New share description. (Default=None)'))
        parser.add_argument('--public', metavar='<public>', help=_('Level of visibility for share. Defines whether other tenants are able to see it or not. '))
        parser.add_argument('--status', metavar='<status>', default=None, help=_('Explicitly update the status of a share (Admin only). Examples include: available, error, creating, deleting, error_deleting.'))
        parser.add_argument('--task-state', metavar='<task-state>', required=False, default=None, help=_('Indicate which task state to assign the share. Options include migration_starting, migration_in_progress, migration_completing, migration_success, migration_error, migration_cancelled, migration_driver_in_progress, migration_driver_phase1_done, data_copying_starting, data_copying_in_progress, data_copying_completing, data_copying_completed, data_copying_cancelled, data_copying_error. '))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share_obj = apiutils.find_resource(share_client.shares, parsed_args.share)
        result = 0
        if parsed_args.property:
            try:
                share_obj.set_metadata(parsed_args.property)
            except Exception as e:
                LOG.error(_("Failed to set share properties '%(properties)s': %(exception)s"), {'properties': parsed_args.property, 'exception': e})
                result += 1
        kwargs = {}
        if parsed_args.name is not None:
            kwargs['display_name'] = parsed_args.name
        if parsed_args.description is not None:
            kwargs['display_description'] = parsed_args.description
        if parsed_args.public is not None:
            kwargs['is_public'] = parsed_args.public
        if kwargs:
            try:
                share_client.shares.update(share_obj.id, **kwargs)
            except Exception as e:
                LOG.error(_('Failed to update share display name, visibility or display description: %s'), e)
                result += 1
        if parsed_args.status:
            try:
                share_obj.reset_state(parsed_args.status)
            except Exception as e:
                LOG.error(_('Failed to set status for the share: %s'), e)
                result += 1
        if parsed_args.task_state:
            try:
                share_obj.reset_task_state(parsed_args.task_state)
            except Exception as e:
                LOG.error(_('Failed to update share task state %s'), e)
                result += 1
        if result > 0:
            raise exceptions.CommandError(_('One or more of the set operations failed'))