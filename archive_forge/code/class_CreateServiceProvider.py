import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class CreateServiceProvider(command.ShowOne):
    _description = _('Create new service provider')

    def get_parser(self, prog_name):
        parser = super(CreateServiceProvider, self).get_parser(prog_name)
        parser.add_argument('service_provider_id', metavar='<name>', help=_('New service provider name (must be unique)'))
        parser.add_argument('--auth-url', metavar='<auth-url>', required=True, help=_('Authentication URL of remote federated service provider (required)'))
        parser.add_argument('--description', metavar='<description>', help=_('New service provider description'))
        parser.add_argument('--service-provider-url', metavar='<sp-url>', required=True, help=_('A service URL where SAML assertions are being sent (required)'))
        enable_service_provider = parser.add_mutually_exclusive_group()
        enable_service_provider.add_argument('--enable', dest='enabled', action='store_true', default=True, help=_('Enable the service provider (default)'))
        enable_service_provider.add_argument('--disable', dest='enabled', action='store_false', help=_('Disable the service provider'))
        return parser

    def take_action(self, parsed_args):
        service_client = self.app.client_manager.identity
        sp = service_client.federation.service_providers.create(id=parsed_args.service_provider_id, auth_url=parsed_args.auth_url, description=parsed_args.description, enabled=parsed_args.enabled, sp_url=parsed_args.service_provider_url)
        sp._info.pop('links', None)
        return zip(*sorted(sp._info.items()))