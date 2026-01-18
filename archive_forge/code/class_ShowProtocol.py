import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ShowProtocol(command.ShowOne):
    _description = _('Display federation protocol details')

    def get_parser(self, prog_name):
        parser = super(ShowProtocol, self).get_parser(prog_name)
        parser.add_argument('federation_protocol', metavar='<federation-protocol>', help=_('Federation protocol to display (name or ID)'))
        parser.add_argument('--identity-provider', metavar='<identity-provider>', required=True, help=_('Identity provider that supports <federation-protocol> (name or ID) (required)'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        protocol = identity_client.federation.protocols.get(parsed_args.identity_provider, parsed_args.federation_protocol)
        info = dict(protocol._info)
        info['mapping'] = info.pop('mapping_id')
        info.pop('links', None)
        return zip(*sorted(info.items()))