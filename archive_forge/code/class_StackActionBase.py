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
class StackActionBase(command.Lister):
    """Stack actions base."""
    log = logging.getLogger(__name__ + '.StackActionBase')

    def _get_parser(self, prog_name, stack_help, wait_help):
        parser = super(StackActionBase, self).get_parser(prog_name)
        parser.add_argument('stack', metavar='<stack>', nargs='+', help=stack_help)
        parser.add_argument('--wait', action='store_true', help=wait_help)
        return parser

    def _take_action(self, parsed_args, action, action_name=None):
        self.log.debug('take_action(%s)', parsed_args)
        heat_client = self.app.client_manager.orchestration
        return _stacks_action(parsed_args, heat_client, action, action_name)