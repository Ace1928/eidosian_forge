import datetime
import logging
from keystoneclient import exceptions as identity_exc
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class CreateTrust(command.ShowOne):
    _description = _('Create new trust')

    def get_parser(self, prog_name):
        parser = super(CreateTrust, self).get_parser(prog_name)
        parser.add_argument('trustor', metavar='<trustor-user>', help=_('User that is delegating authorization (name or ID)'))
        parser.add_argument('trustee', metavar='<trustee-user>', help=_('User that is assuming authorization (name or ID)'))
        parser.add_argument('--project', metavar='<project>', required=True, help=_('Project being delegated (name or ID) (required)'))
        parser.add_argument('--role', metavar='<role>', action='append', default=[], help=_('Roles to authorize (name or ID) (repeat option to set multiple values, required)'), required=True)
        parser.add_argument('--impersonate', dest='impersonate', action='store_true', default=False, help=_('Tokens generated from the trust will represent <trustor> (defaults to False)'))
        parser.add_argument('--expiration', metavar='<expiration>', help=_('Sets an expiration date for the trust (format of YYYY-mm-ddTHH:MM:SS)'))
        common.add_project_domain_option_to_parser(parser)
        parser.add_argument('--trustor-domain', metavar='<trustor-domain>', help=_('Domain that contains <trustor> (name or ID)'))
        parser.add_argument('--trustee-domain', metavar='<trustee-domain>', help=_('Domain that contains <trustee> (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        trustor_id = common.find_user(identity_client, parsed_args.trustor, parsed_args.trustor_domain).id
        trustee_id = common.find_user(identity_client, parsed_args.trustee, parsed_args.trustee_domain).id
        project_id = common.find_project(identity_client, parsed_args.project, parsed_args.project_domain).id
        role_ids = []
        for role in parsed_args.role:
            try:
                role_id = utils.find_resource(identity_client.roles, role).id
            except identity_exc.Forbidden:
                role_id = role
            role_ids.append(role_id)
        expires_at = None
        if parsed_args.expiration:
            expires_at = datetime.datetime.strptime(parsed_args.expiration, '%Y-%m-%dT%H:%M:%S')
        trust = identity_client.trusts.create(trustee_id, trustor_id, impersonation=parsed_args.impersonate, project=project_id, role_ids=role_ids, expires_at=expires_at)
        trust._info.pop('roles_links', None)
        trust._info.pop('links', None)
        roles = trust._info.pop('roles')
        msg = ' '.join((r['name'] for r in roles))
        trust._info['roles'] = msg
        return zip(*sorted(trust._info.items()))