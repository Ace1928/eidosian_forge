import datetime
import json
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class ListApplicationCredential(command.Lister):
    _description = _('List application credentials')

    def get_parser(self, prog_name):
        parser = super(ListApplicationCredential, self).get_parser(prog_name)
        parser.add_argument('--user', metavar='<user>', help=_('User whose application credentials to list (name or ID)'))
        common.add_user_domain_option_to_parser(parser)
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        if parsed_args.user:
            user_id = common.find_user(identity_client, parsed_args.user, parsed_args.user_domain).id
        else:
            user_id = None
        columns = ('ID', 'Name', 'Project ID', 'Description', 'Expires At')
        data = identity_client.application_credentials.list(user=user_id)
        return (columns, (utils.get_item_properties(s, columns, formatters={}) for s in data))