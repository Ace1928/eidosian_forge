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
def _stacks_action(parsed_args, heat_client, action, action_name=None):
    rows = []
    columns = ['ID', 'Stack Name', 'Stack Status', 'Creation Time', 'Updated Time']
    for stack in parsed_args.stack:
        data = _stack_action(stack, parsed_args, heat_client, action, action_name)
        rows += [utils.get_dict_properties(data.to_dict(), columns)]
    return (columns, rows)