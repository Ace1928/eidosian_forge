import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ListEC2Creds(command.Lister):
    _description = _('List EC2 credentials')

    def get_parser(self, prog_name):
        parser = super(ListEC2Creds, self).get_parser(prog_name)
        parser.add_argument('--user', metavar='<user>', help=_('Filter list by user (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        if parsed_args.user:
            user = utils.find_resource(identity_client.users, parsed_args.user).id
        else:
            user = self.app.client_manager.auth_ref.user_id
        columns = ('access', 'secret', 'tenant_id', 'user_id')
        column_headers = ('Access', 'Secret', 'Project ID', 'User ID')
        data = identity_client.ec2.list(user)
        return (column_headers, (utils.get_item_properties(s, columns, formatters={}) for s in data))