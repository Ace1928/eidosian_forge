from osc_lib.command import command
from osc_lib import exceptions
from openstackclient.i18n import _
class RevokeToken(command.Command):
    _description = _('Revoke existing token')

    def get_parser(self, prog_name):
        parser = super(RevokeToken, self).get_parser(prog_name)
        parser.add_argument('token', metavar='<token>', help=_('Token to be deleted'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        identity_client.tokens.delete(parsed_args.token)