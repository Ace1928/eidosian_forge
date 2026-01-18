import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class ShowService(command.ShowOne):
    _description = _('Display service details')

    def get_parser(self, prog_name):
        parser = super(ShowService, self).get_parser(prog_name)
        parser.add_argument('service', metavar='<service>', help=_('Service to display (type, name or ID)'))
        parser.add_argument('--catalog', action='store_true', default=False, help=_('Show service catalog information'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        auth_ref = self.app.client_manager.auth_ref
        if parsed_args.catalog:
            endpoints = auth_ref.service_catalog.get_endpoints(service_type=parsed_args.service)
            for service, service_endpoints in endpoints.items():
                if service_endpoints:
                    info = {'type': service}
                    info.update(service_endpoints[0])
                    return zip(*sorted(info.items()))
            msg = _("No service catalog with a type, name or ID of '%s' exists.") % parsed_args.service
            raise exceptions.CommandError(msg)
        else:
            service = common.find_service(identity_client, parsed_args.service)
            info = {}
            info.update(service._info)
            return zip(*sorted(info.items()))