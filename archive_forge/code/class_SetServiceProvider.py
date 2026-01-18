import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class SetServiceProvider(command.Command):
    _description = _('Set service provider properties')

    def get_parser(self, prog_name):
        parser = super(SetServiceProvider, self).get_parser(prog_name)
        parser.add_argument('service_provider', metavar='<service-provider>', help=_('Service provider to modify'))
        parser.add_argument('--auth-url', metavar='<auth-url>', help=_('New Authentication URL of remote federated service provider'))
        parser.add_argument('--description', metavar='<description>', help=_('New service provider description'))
        parser.add_argument('--service-provider-url', metavar='<sp-url>', help=_('New service provider URL, where SAML assertions are sent'))
        enable_service_provider = parser.add_mutually_exclusive_group()
        enable_service_provider.add_argument('--enable', action='store_true', help=_('Enable the service provider'))
        enable_service_provider.add_argument('--disable', action='store_true', help=_('Disable the service provider'))
        return parser

    def take_action(self, parsed_args):
        federation_client = self.app.client_manager.identity.federation
        enabled = None
        if parsed_args.enable is True:
            enabled = True
        elif parsed_args.disable is True:
            enabled = False
        federation_client.service_providers.update(parsed_args.service_provider, enabled=enabled, description=parsed_args.description, auth_url=parsed_args.auth_url, sp_url=parsed_args.service_provider_url)