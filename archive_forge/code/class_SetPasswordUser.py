import copy
import logging
from keystoneauth1 import exceptions as ks_exc
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class SetPasswordUser(command.Command):
    _description = _('Change current user password')
    required_scope = False

    def get_parser(self, prog_name):
        parser = super(SetPasswordUser, self).get_parser(prog_name)
        parser.add_argument('--password', metavar='<new-password>', help=_('New user password'))
        parser.add_argument('--original-password', metavar='<original-password>', help=_('Original user password'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        current_password = parsed_args.original_password
        if current_password is None:
            current_password = utils.get_password(self.app.stdin, prompt='Current Password:', confirm=False)
        password = parsed_args.password
        if password is None:
            password = utils.get_password(self.app.stdin, prompt='New Password:')
        if '' == password:
            LOG.warning(_('No password was supplied, authentication will fail when a user does not have a password.'))
        identity_client.users.update_password(current_password, password)