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
class ResourceMetadata(format_utils.JsonFormat):
    """Show resource metadata"""
    log = logging.getLogger(__name__ + '.ResourceMetadata')

    def get_parser(self, prog_name):
        parser = super(ResourceMetadata, self).get_parser(prog_name)
        parser.add_argument('stack', metavar='<stack>', help=_('Stack to display (name or ID)'))
        parser.add_argument('resource', metavar='<resource>', help=_('Name of the resource to show the metadata for'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        heat_client = self.app.client_manager.orchestration
        return _resource_metadata(heat_client, parsed_args)