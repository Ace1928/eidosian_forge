import functools
import logging
from cliff import columns as cliff_columns
from keystoneauth1 import exceptions as ks_exc
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class DeleteUser(command.Command):
    _description = _('Delete user(s)')

    def get_parser(self, prog_name):
        parser = super(DeleteUser, self).get_parser(prog_name)
        parser.add_argument('users', metavar='<user>', nargs='+', help=_('User(s) to delete (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        errors = 0
        for user in parsed_args.users:
            try:
                user_obj = utils.find_resource(identity_client.users, user)
                identity_client.users.delete(user_obj.id)
            except Exception as e:
                errors += 1
                LOG.error(_("Failed to delete user with name or ID '%(user)s': %(e)s"), {'user': user, 'e': e})
        if errors > 0:
            total = len(parsed_args.users)
            msg = _('%(errors)s of %(total)s users failed to delete.') % {'errors': errors, 'total': total}
            raise exceptions.CommandError(msg)