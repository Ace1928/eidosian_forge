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
def _hook_clear(args, heat_client):
    """Clear resource hooks on a given stack."""
    if args.pre_create:
        hook_type = 'pre-create'
    elif args.pre_update:
        hook_type = 'pre-update'
    elif args.pre_delete:
        hook_type = 'pre-delete'
    else:
        hook_type = hook_utils.get_hook_type_via_status(heat_client, args.stack)
    for hook_string in args.hook:
        hook = [b for b in hook_string.split('/') if b]
        resource_pattern = hook[-1]
        stack_id = args.stack
        hook_utils.clear_wildcard_hooks(heat_client, stack_id, hook[:-1], hook_type, resource_pattern)