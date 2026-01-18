from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class AuthorizeRequestToken(command.ShowOne):
    _description = _('Authorize a request token')

    def get_parser(self, prog_name):
        parser = super(AuthorizeRequestToken, self).get_parser(prog_name)
        parser.add_argument('--request-key', metavar='<request-key>', required=True, help=_('Request token to authorize (ID only) (required)'))
        parser.add_argument('--role', metavar='<role>', action='append', default=[], required=True, help=_('Roles to authorize (name or ID) (repeat option to set multiple values) (required)'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        roles = []
        for role in parsed_args.role:
            role_id = utils.find_resource(identity_client.roles, role).id
            roles.append(role_id)
        verifier_pin = identity_client.oauth1.request_tokens.authorize(parsed_args.request_key, roles)
        return zip(*sorted(verifier_pin._info.items()))