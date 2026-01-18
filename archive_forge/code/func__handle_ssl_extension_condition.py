from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import compare_complex_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _handle_ssl_extension_condition(self, action, item):
    options = ['server_name_is_any', 'server_name_is_not_any', 'server_name_contains', 'server_name_begins_with_any', 'server_name_begins_not_with_any', 'server_name_ends_with_any', 'server_name_ends_not_with_any']
    action['type'] = 'ssl_extension'
    if 'server_name_is_any' in item and item['server_name_is_any'] is not None:
        if isinstance(item['server_name_is_any'], list):
            values = item['server_name_is_any']
        else:
            values = [item['server_name_is_any']]
        action.update(dict(equals=True, serverName=True, values=values))
    if 'server_name_is_not_any' in item and item['server_name_is_not_any'] is not None:
        if isinstance(item['server_name_is_not_any'], list):
            values = item['server_name_is_not_any']
        else:
            values = [item['server_name_is_not_any']]
        action.update({'equals': True, 'serverName': True, 'not': True, 'values': values})
    if 'server_name_begins_with_any' in item and item['server_name_begins_with_any'] is not None:
        if isinstance(item['server_name_begins_with_any'], list):
            values = item['server_name_begins_with_any']
        else:
            values = [item['server_name_begins_with_any']]
        action.update(dict(serverName=True, startsWith=True, values=values))
    if 'server_name_begins_not_with_any' in item and item['server_name_begins_not_with_any'] is not None:
        if isinstance(item['server_name_begins_not_with_any'], list):
            values = item['server_name_begins_not_with_any']
        else:
            values = [item['server_name_begins_not_with_any']]
        action.update({'serverName': True, 'startsWith': True, 'not': True, 'values': values})
    if 'server_name_ends_with_any' in item and item['server_name_ends_with_any'] is not None:
        if isinstance(item['server_name_ends_with_any'], list):
            values = item['server_name_ends_with_any']
        else:
            values = [item['server_name_ends_with_any']]
        action.update(dict(serverName=True, endsWith=True, values=values))
    if 'server_name_ends_not_with_any' in item and item['server_name_ends_not_with_any'] is not None:
        if isinstance(item['server_name_ends_not_with_any'], list):
            values = item['server_name_ends_not_with_any']
        else:
            values = [item['server_name_ends_not_with_any']]
        action.update({'serverName': True, 'endsWith': True, 'not': True, 'values': values})
    if 'server_name_contains' in item and item['server_name_contains'] is not None:
        if isinstance(item['server_name_contains'], list):
            values = item['server_name_contains']
        else:
            values = [item['server_name_contains']]
        action.update({'serverName': True, 'contains': True, 'values': values})
    if 'event' not in item:
        raise F5ModuleError("An 'event' must be specified when the 'ssl_extension' condition is used.")
    elif 'ssl_client_hello' in item['event']:
        action.update(dict(sslClientHello=True))
    elif 'ssl_server_hello' in item['event']:
        action.update(dict(sslServerHello=True))