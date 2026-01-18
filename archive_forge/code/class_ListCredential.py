import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class ListCredential(command.Lister):
    _description = _('List credentials')

    def get_parser(self, prog_name):
        parser = super(ListCredential, self).get_parser(prog_name)
        parser.add_argument('--user', metavar='<user>', help=_('Filter credentials by <user> (name or ID)'))
        common.add_user_domain_option_to_parser(parser)
        parser.add_argument('--type', metavar='<type>', help=_('Filter credentials by type: cert, ec2, totp and so on'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        kwargs = {}
        if parsed_args.user:
            user_id = common.find_user(identity_client, parsed_args.user, parsed_args.user_domain).id
            kwargs['user_id'] = user_id
        if parsed_args.type:
            kwargs['type'] = parsed_args.type
        columns = ('ID', 'Type', 'User ID', 'Blob', 'Project ID')
        column_headers = ('ID', 'Type', 'User ID', 'Data', 'Project ID')
        data = self.app.client_manager.identity.credentials.list(**kwargs)
        return (column_headers, (utils.get_item_properties(s, columns, formatters={}) for s in data))