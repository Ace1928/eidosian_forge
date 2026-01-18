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
def _show_stack(heat_client, stack_id, format='', short=False, resolve_outputs=True):
    try:
        _resolve_outputs = not short and resolve_outputs
        data = heat_client.stacks.get(stack_id=stack_id, resolve_outputs=_resolve_outputs)
    except heat_exc.HTTPNotFound:
        raise exc.CommandError('Stack not found: %s' % stack_id)
    else:
        columns = ['id', 'stack_name', 'description', 'creation_time', 'updated_time', 'stack_status', 'stack_status_reason']
        if not short:
            columns.append('parameters')
            if _resolve_outputs:
                columns.append('outputs')
            columns.append('links')
            exclude_columns = ('template_description',)
            for key in data.to_dict():
                if key not in columns and key not in exclude_columns:
                    columns.append(key)
        formatters = {}
        complex_formatter = None
        if format in 'table':
            complex_formatter = heat_utils.yaml_formatter
        elif format in ('shell', 'value', 'html'):
            complex_formatter = heat_utils.json_formatter
        if complex_formatter:
            formatters['parameters'] = complex_formatter
            formatters['outputs'] = complex_formatter
            formatters['links'] = complex_formatter
            formatters['tags'] = complex_formatter
        return (columns, utils.get_item_properties(data, columns, formatters=formatters))