import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class CreateEndpoint(command.ShowOne):
    _description = _('Create new endpoint')

    def get_parser(self, prog_name):
        parser = super(CreateEndpoint, self).get_parser(prog_name)
        parser.add_argument('service', metavar='<service>', help=_('Service to be associated with new endpoint (name or ID)'))
        parser.add_argument('--publicurl', metavar='<url>', required=True, help=_('New endpoint public URL (required)'))
        parser.add_argument('--adminurl', metavar='<url>', help=_('New endpoint admin URL'))
        parser.add_argument('--internalurl', metavar='<url>', help=_('New endpoint internal URL'))
        parser.add_argument('--region', metavar='<region-id>', help=_('New endpoint region ID'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        service = common.find_service(identity_client, parsed_args.service)
        endpoint = identity_client.endpoints.create(parsed_args.region, service.id, parsed_args.publicurl, parsed_args.adminurl, parsed_args.internalurl)
        info = {}
        info.update(endpoint._info)
        info['service_name'] = service.name
        info['service_type'] = service.type
        return zip(*sorted(info.items()))