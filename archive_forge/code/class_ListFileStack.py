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
class ListFileStack(format_utils.YamlFormat):
    """Show a stack's files map."""
    log = logging.getLogger(__name__)

    def get_parser(self, prog_name):
        parser = super(ListFileStack, self).get_parser(prog_name)
        parser.add_argument('stack', metavar='<NAME or ID>', help=_('Name or ID of stack to query'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        client = self.app.client_manager.orchestration
        try:
            files = client.stacks.files(stack_id=parsed_args.stack)
        except heat_exc.HTTPNotFound:
            msg = _('Stack not found: %s') % parsed_args.stack
            raise exc.CommandError(msg)
        return (['files'], [files])