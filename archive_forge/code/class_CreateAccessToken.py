from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class CreateAccessToken(command.ShowOne):
    _description = _('Create an access token')

    def get_parser(self, prog_name):
        parser = super(CreateAccessToken, self).get_parser(prog_name)
        parser.add_argument('--consumer-key', metavar='<consumer-key>', help=_('Consumer key (required)'), required=True)
        parser.add_argument('--consumer-secret', metavar='<consumer-secret>', help=_('Consumer secret (required)'), required=True)
        parser.add_argument('--request-key', metavar='<request-key>', help=_('Request token to exchange for access token (required)'), required=True)
        parser.add_argument('--request-secret', metavar='<request-secret>', help=_('Secret associated with <request-key> (required)'), required=True)
        parser.add_argument('--verifier', metavar='<verifier>', help=_('Verifier associated with <request-key> (required)'), required=True)
        return parser

    def take_action(self, parsed_args):
        token_client = self.app.client_manager.identity.oauth1.access_tokens
        access_token = token_client.create(parsed_args.consumer_key, parsed_args.consumer_secret, parsed_args.request_key, parsed_args.request_secret, parsed_args.verifier)
        return zip(*sorted(access_token._info.items()))