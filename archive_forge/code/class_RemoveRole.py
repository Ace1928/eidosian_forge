import logging
from keystoneauth1 import exceptions as ks_exc
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class RemoveRole(command.Command):
    _description = _('Remove role from project : user')

    def get_parser(self, prog_name):
        parser = super(RemoveRole, self).get_parser(prog_name)
        parser.add_argument('role', metavar='<role>', help=_('Role to remove (name or ID)'))
        parser.add_argument('--project', metavar='<project>', required=True, help=_('Include <project> (name or ID)'))
        parser.add_argument('--user', metavar='<user>', required=True, help=_('Include <user> (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        role = utils.find_resource(identity_client.roles, parsed_args.role)
        project = utils.find_resource(identity_client.tenants, parsed_args.project)
        user = utils.find_resource(identity_client.users, parsed_args.user)
        identity_client.roles.remove_user_role(user.id, role.id, project.id)