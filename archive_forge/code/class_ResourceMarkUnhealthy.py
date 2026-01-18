import logging
from osc_lib.command import command
from osc_lib import exceptions as exc
from osc_lib.i18n import _
from osc_lib import utils
from oslo_serialization import jsonutils
from urllib import request
from heatclient.common import format_utils
from heatclient.common import utils as heat_utils
from heatclient import exc as heat_exc
class ResourceMarkUnhealthy(command.Command):
    """Set resource's health."""
    log = logging.getLogger(__name__ + '.ResourceMarkUnhealthy')

    def get_parser(self, prog_name):
        parser = super(ResourceMarkUnhealthy, self).get_parser(prog_name)
        parser.add_argument('stack', metavar='<stack>', help=_('Name or ID of stack the resource belongs to'))
        parser.add_argument('resource', metavar='<resource>', help=_('Name of the resource'))
        parser.add_argument('reason', default='', nargs='?', help=_('Reason for state change'))
        parser.add_argument('--reset', default=False, action='store_true', help=_('Set the resource as healthy'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        heat_client = self.app.client_manager.orchestration
        fields = {'stack_id': parsed_args.stack, 'resource_name': parsed_args.resource, 'mark_unhealthy': not parsed_args.reset, 'resource_status_reason': parsed_args.reason}
        try:
            heat_client.resources.mark_unhealthy(**fields)
        except heat_exc.HTTPNotFound:
            raise exc.CommandError(_('Stack or resource not found: %(id)s %(resource)s') % {'id': parsed_args.stack, 'resource': parsed_args.resource})