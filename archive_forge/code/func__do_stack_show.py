import logging
import sys
from oslo_serialization import jsonutils
from oslo_utils import strutils
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient.common import deployment_utils
from heatclient.common import event_utils
from heatclient.common import hook_utils
from heatclient.common import http
from heatclient.common import template_format
from heatclient.common import template_utils
from heatclient.common import utils
import heatclient.exc as exc
def _do_stack_show(hc, fields):
    try:
        stack = hc.stacks.get(**fields)
    except exc.HTTPNotFound:
        raise exc.CommandError(_('Stack not found: %s') % fields.get('stack_id'))
    else:
        formatters = {'description': utils.text_wrap_formatter, 'template_description': utils.text_wrap_formatter, 'stack_status_reason': utils.text_wrap_formatter, 'parameters': utils.json_formatter, 'outputs': utils.json_formatter, 'links': utils.link_formatter, 'tags': utils.json_formatter}
        utils.print_dict(stack.to_dict(), formatters=formatters)