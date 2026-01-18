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
class ResourceSignal(command.Command):
    """Signal a resource with optional data."""
    log = logging.getLogger(__name__ + '.ResourceSignal')

    def get_parser(self, prog_name):
        parser = super(ResourceSignal, self).get_parser(prog_name)
        parser.add_argument('stack', metavar='<stack>', help=_('Name or ID of stack the resource belongs to'))
        parser.add_argument('resource', metavar='<resource>', help=_('Name of the resoure to signal'))
        parser.add_argument('--data', metavar='<data>', help=_('JSON Data to send to the signal handler'))
        parser.add_argument('--data-file', metavar='<data-file>', help=_('File containing JSON data to send to the signal handler'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        heat_client = self.app.client_manager.orchestration
        return _resource_signal(heat_client, parsed_args)