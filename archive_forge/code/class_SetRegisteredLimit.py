import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as common_utils
class SetRegisteredLimit(command.ShowOne):
    _description = _('Update information about a registered limit')

    def get_parser(self, prog_name):
        parser = super(SetRegisteredLimit, self).get_parser(prog_name)
        parser.add_argument('registered_limit_id', metavar='<registered-limit-id>', help=_('Registered limit to update (ID)'))
        parser.add_argument('--service', metavar='<service>', help=_('Service to be updated responsible for the resource to limit. Either --service, --resource-name or --region must be different than existing value otherwise it will be duplicate entry'))
        parser.add_argument('--resource-name', metavar='<resource-name>', help=_('Resource to be updated responsible for the resource to limit. Either --service, --resource-name or --region must be different than existing value otherwise it will be duplicate entry'))
        parser.add_argument('--default-limit', metavar='<default-limit>', type=int, help=_('The default limit for the resources to assume'))
        parser.add_argument('--description', metavar='<description>', help=_('Description to update of the registered limit'))
        parser.add_argument('--region', metavar='<region>', help=_('Region for the registered limit to affect. Either --service, --resource-name or --region must be different than existing value otherwise it will be duplicate entry'))
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
        registered_limit = identity_client.registered_limits.update(parsed_args.registered_limit_id, service=service, resource_name=parsed_args.resource_name, default_limit=parsed_args.default_limit, description=parsed_args.description, region=region)
        registered_limit._info.pop('links', None)
        return zip(*sorted(registered_limit._info.items()))