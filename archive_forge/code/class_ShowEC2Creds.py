import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ShowEC2Creds(command.ShowOne):
    _description = _('Display EC2 credentials details')

    def get_parser(self, prog_name):
        parser = super(ShowEC2Creds, self).get_parser(prog_name)
        parser.add_argument('access_key', metavar='<access-key>', help=_('Credentials access key'))
        parser.add_argument('--user', metavar='<user>', help=_('Show credentials for user (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        if parsed_args.user:
            user = utils.find_resource(identity_client.users, parsed_args.user).id
        else:
            user = self.app.client_manager.auth_ref.user_id
        creds = identity_client.ec2.get(user, parsed_args.access_key)
        info = {}
        info.update(creds._info)
        if 'tenant_id' in info:
            info.update({'project_id': info.pop('tenant_id')})
        return zip(*sorted(info.items()))