import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class ShowEndpoint(command.ShowOne):
    _description = _('Display endpoint details')

    def get_parser(self, prog_name):
        parser = super(ShowEndpoint, self).get_parser(prog_name)
        parser.add_argument('endpoint_or_service', metavar='<endpoint>', help=_('Endpoint to display (endpoint ID, service ID, service name, service type)'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        data = identity_client.endpoints.list()
        match = None
        for ep in data:
            if ep.id == parsed_args.endpoint_or_service:
                match = ep
                service = common.find_service(identity_client, ep.service_id)
        if match is None:
            service = common.find_service(identity_client, parsed_args.endpoint_or_service)
            for ep in data:
                if ep.service_id == service.id:
                    match = ep
        if match is None:
            return None
        info = {}
        info.update(match._info)
        info['service_name'] = service.name
        info['service_type'] = service.type
        return zip(*sorted(info.items()))