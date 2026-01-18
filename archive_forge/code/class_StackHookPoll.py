import logging
import sys
from osc_lib.command import command
from osc_lib import exceptions as exc
from osc_lib import utils
from oslo_serialization import jsonutils
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient.common import event_utils
from heatclient.common import format_utils
from heatclient.common import hook_utils
from heatclient.common import http
from heatclient.common import template_utils
from heatclient.common import utils as heat_utils
from heatclient import exc as heat_exc
class StackHookPoll(command.Lister):
    """List resources with pending hook for a stack."""
    log = logging.getLogger(__name__ + '.StackHookPoll')

    def get_parser(self, prog_name):
        parser = super(StackHookPoll, self).get_parser(prog_name)
        parser.add_argument('stack', metavar='<stack>', help=_('Stack to display (name or ID)'))
        parser.add_argument('--nested-depth', metavar='<nested-depth>', help=_('Depth of nested stacks from which to display hooks'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        heat_client = self.app.client_manager.orchestration
        return _hook_poll(parsed_args, heat_client)