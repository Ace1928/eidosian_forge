import datetime
import json
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class CreateApplicationCredential(command.ShowOne):
    _description = _('Create new application credential')

    def get_parser(self, prog_name):
        parser = super(CreateApplicationCredential, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<name>', help=_('Name of the application credential'))
        parser.add_argument('--secret', metavar='<secret>', help=_('Secret to use for authentication (if not provided, one will be generated)'))
        parser.add_argument('--role', metavar='<role>', action='append', default=[], help=_('Roles to authorize (name or ID) (repeat option to set multiple values)'))
        parser.add_argument('--expiration', metavar='<expiration>', help=_('Sets an expiration date for the application credential, format of YYYY-mm-ddTHH:MM:SS (if not provided, the application credential will not expire)'))
        parser.add_argument('--description', metavar='<description>', help=_('Application credential description'))
        parser.add_argument('--unrestricted', action='store_true', help=_('Enable application credential to create and delete other application credentials and trusts (this is potentially dangerous behavior and is disabled by default)'))
        parser.add_argument('--restricted', action='store_true', help=_('Prohibit application credential from creating and deleting other application credentials and trusts (this is the default behavior)'))
        parser.add_argument('--access-rules', metavar='<access-rules>', help=_('Either a string or file path containing a JSON-formatted list of access rules, each containing a request method, path, and service, for example \'[{"method": "GET", "path": "/v2.1/servers", "service": "compute"}]\''))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        role_ids = []
        for role in parsed_args.role:
            role_id = common._get_token_resource(identity_client, 'roles', role)
            role_ids.append(role_id)
        expires_at = None
        if parsed_args.expiration:
            expires_at = datetime.datetime.strptime(parsed_args.expiration, '%Y-%m-%dT%H:%M:%S')
        if parsed_args.restricted:
            unrestricted = False
        else:
            unrestricted = parsed_args.unrestricted
        if parsed_args.access_rules:
            try:
                access_rules = json.loads(parsed_args.access_rules)
            except ValueError:
                try:
                    with open(parsed_args.access_rules) as f:
                        access_rules = json.load(f)
                except IOError:
                    msg = _('Access rules is not valid JSON string or file does not exist.')
                    raise exceptions.CommandError(msg)
        else:
            access_rules = None
        app_cred_manager = identity_client.application_credentials
        application_credential = app_cred_manager.create(parsed_args.name, roles=role_ids, expires_at=expires_at, description=parsed_args.description, secret=parsed_args.secret, unrestricted=unrestricted, access_rules=access_rules)
        application_credential._info.pop('links', None)
        roles = application_credential._info.pop('roles')
        msg = ' '.join((r['name'] for r in roles))
        application_credential._info['roles'] = msg
        return zip(*sorted(application_credential._info.items()))