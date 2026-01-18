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
class SuspendStack(StackActionBase):
    """Suspend a stack."""
    log = logging.getLogger(__name__ + '.SuspendStack')

    def get_parser(self, prog_name):
        return self._get_parser(prog_name, _('Stack(s) to suspend (name or ID)'), _('Wait for suspend to complete'))

    def take_action(self, parsed_args):
        return self._take_action(parsed_args, self.app.client_manager.orchestration.actions.suspend, 'SUSPEND')