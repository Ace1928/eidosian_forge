import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class CreateEC2Creds(command.ShowOne):
    _description = _('Create EC2 credentials')

    def get_parser(self, prog_name):
        parser = super(CreateEC2Creds, self).get_parser(prog_name)
        parser.add_argument('--project', metavar='<project>', help=_('Create credentials in project (name or ID; default: current authenticated project)'))
        parser.add_argument('--user', metavar='<user>', help=_('Create credentials for user (name or ID; default: current authenticated user)'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        if parsed_args.project:
            project = utils.find_resource(identity_client.tenants, parsed_args.project).id
        else:
            project = self.app.client_manager.auth_ref.project_id
        if parsed_args.user:
            user = utils.find_resource(identity_client.users, parsed_args.user).id
        else:
            user = self.app.client_manager.auth_ref.user_id
        creds = identity_client.ec2.create(user, project)
        info = {}
        info.update(creds._info)
        if 'tenant_id' in info:
            info.update({'project_id': info.pop('tenant_id')})
        return zip(*sorted(info.items()))