import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as common_utils
class ListRegisteredLimit(command.Lister):
    _description = _('List registered limits')

    def get_parser(self, prog_name):
        parser = super(ListRegisteredLimit, self).get_parser(prog_name)
        parser.add_argument('--service', metavar='<service>', help=_('Service responsible for the resource to limit'))
        parser.add_argument('--resource-name', metavar='<resource-name>', dest='resource_name', help=_('The name of the resource to limit'))
        parser.add_argument('--region', metavar='<region>', help=_('Region for the limit to affect.'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        service = None
        if parsed_args.service:
            service = common_utils.find_service(identity_client, parsed_args.service)
        region = None
        if parsed_args.region:
            val = getattr(parsed_args, 'region', None)
            if 'None' not in val:
                region = common_utils.get_resource(identity_client.regions, parsed_args.region)
        registered_limits = identity_client.registered_limits.list(service=service, resource_name=parsed_args.resource_name, region=region)
        columns = ('ID', 'Service ID', 'Resource Name', 'Default Limit', 'Description', 'Region ID')
        return (columns, (utils.get_item_properties(s, columns) for s in registered_limits))