import datetime
import logging
from keystoneclient import exceptions as identity_exc
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class ListTrust(command.Lister):
    _description = _('List trusts')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--trustor', metavar='<trustor-user>', help=_('Trustor user to filter (name or ID)'))
        parser.add_argument('--trustee', metavar='<trustee-user>', help=_('Trustee user to filter (name or ID)'))
        parser.add_argument('--trustor-domain', metavar='<trustor-domain>', help=_('Domain that contains <trustor> (name or ID)'))
        parser.add_argument('--trustee-domain', metavar='<trustee-domain>', help=_('Domain that contains <trustee> (name or ID)'))
        parser.add_argument('--auth-user', action='store_true', dest='authuser', help=_('Only list trusts related to the authenticated user'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        auth_ref = self.app.client_manager.auth_ref
        if parsed_args.authuser and any([parsed_args.trustor, parsed_args.trustor_domain, parsed_args.trustee, parsed_args.trustee_domain]):
            msg = _('--authuser cannot be used with --trustee or --trustor')
            raise exceptions.CommandError(msg)
        if parsed_args.trustee_domain and (not parsed_args.trustee):
            msg = _('Using --trustee-domain mandates the use of --trustee')
            raise exceptions.CommandError(msg)
        if parsed_args.trustor_domain and (not parsed_args.trustor):
            msg = _('Using --trustor-domain mandates the use of --trustor')
            raise exceptions.CommandError(msg)
        if parsed_args.authuser:
            if auth_ref:
                user = common.find_user(identity_client, auth_ref.user_id)
                data1 = identity_client.trusts.list(trustor_user=user)
                data2 = identity_client.trusts.list(trustee_user=user)
                data = set(data1 + data2)
        else:
            trustor = None
            if parsed_args.trustor:
                trustor = common.find_user(identity_client, parsed_args.trustor, parsed_args.trustor_domain)
            trustee = None
            if parsed_args.trustee:
                trustee = common.find_user(identity_client, parsed_args.trustor, parsed_args.trustor_domain)
            data = self.app.client_manager.identity.trusts.list(trustor_user=trustor, trustee_user=trustee)
        columns = ('ID', 'Expires At', 'Impersonation', 'Project ID', 'Trustee User ID', 'Trustor User ID')
        return (columns, (utils.get_item_properties(s, columns, formatters={}) for s in data))