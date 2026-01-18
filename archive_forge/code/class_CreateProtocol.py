import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class CreateProtocol(command.ShowOne):
    _description = _('Create new federation protocol')

    def get_parser(self, prog_name):
        parser = super(CreateProtocol, self).get_parser(prog_name)
        parser.add_argument('federation_protocol', metavar='<name>', help=_('New federation protocol name (must be unique per identity provider)'))
        parser.add_argument('--identity-provider', metavar='<identity-provider>', required=True, help=_('Identity provider that will support the new federation  protocol (name or ID) (required)'))
        parser.add_argument('--mapping', metavar='<mapping>', required=True, help=_('Mapping that is to be used (name or ID) (required)'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        protocol = identity_client.federation.protocols.create(protocol_id=parsed_args.federation_protocol, identity_provider=parsed_args.identity_provider, mapping=parsed_args.mapping)
        info = dict(protocol._info)
        info['identity_provider'] = parsed_args.identity_provider
        info['mapping'] = info.pop('mapping_id')
        info.pop('links', None)
        return zip(*sorted(info.items()))