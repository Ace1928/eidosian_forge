import functools
import logging
from cliff import columns as cliff_columns
from keystoneauth1 import exceptions as ks_exc
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class SetUser(command.Command):
    _description = _('Set user properties')

    def get_parser(self, prog_name):
        parser = super(SetUser, self).get_parser(prog_name)
        parser.add_argument('user', metavar='<user>', help=_('User to modify (name or ID)'))
        parser.add_argument('--name', metavar='<name>', help=_('Set user name'))
        parser.add_argument('--project', metavar='<project>', help=_('Set default project (name or ID)'))
        parser.add_argument('--password', metavar='<user-password>', help=_('Set user password'))
        parser.add_argument('--password-prompt', dest='password_prompt', action='store_true', help=_('Prompt interactively for password'))
        parser.add_argument('--email', metavar='<email-address>', help=_('Set user email address'))
        enable_group = parser.add_mutually_exclusive_group()
        enable_group.add_argument('--enable', action='store_true', help=_('Enable user (default)'))
        enable_group.add_argument('--disable', action='store_true', help=_('Disable user'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        if parsed_args.password_prompt:
            parsed_args.password = utils.get_password(self.app.stdin)
        if '' == parsed_args.password:
            LOG.warning(_('No password was supplied, authentication will fail when a user does not have a password.'))
        user = utils.find_resource(identity_client.users, parsed_args.user)
        if parsed_args.password:
            identity_client.users.update_password(user.id, parsed_args.password)
        if parsed_args.project:
            project = utils.find_resource(identity_client.tenants, parsed_args.project)
            identity_client.users.update_tenant(user.id, project.id)
        kwargs = {}
        if parsed_args.name:
            kwargs['name'] = parsed_args.name
        if parsed_args.email:
            kwargs['email'] = parsed_args.email
        kwargs['enabled'] = user.enabled
        if parsed_args.enable:
            kwargs['enabled'] = True
        if parsed_args.disable:
            kwargs['enabled'] = False
        identity_client.users.update(user.id, **kwargs)