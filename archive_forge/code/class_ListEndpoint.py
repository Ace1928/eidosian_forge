import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class ListEndpoint(command.Lister):
    _description = _('List endpoints')

    def get_parser(self, prog_name):
        parser = super(ListEndpoint, self).get_parser(prog_name)
        parser.add_argument('--long', action='store_true', default=False, help=_('List additional fields in output'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        if parsed_args.long:
            columns = ('ID', 'Region', 'Service Name', 'Service Type', 'PublicURL', 'AdminURL', 'InternalURL')
        else:
            columns = ('ID', 'Region', 'Service Name', 'Service Type')
        data = identity_client.endpoints.list()
        for ep in data:
            service = common.find_service(identity_client, ep.service_id)
            ep.service_name = service.name
            ep.service_type = service.type
        return (columns, (utils.get_item_properties(s, columns, formatters={}) for s in data))