from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import compare_complex_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _handle_http_header_condition(self, action, item):
    action['type'] = 'http_header'
    options = ['header_is_any']
    event_map = dict(proxy_connect='proxyConnect', proxy_request='proxyRequest', proxy_response='proxyResponse', request='request', response='response')
    if 'header_name' not in item:
        raise F5ModuleError("An 'header_name' must be specified when the 'http_header' condition is used.")
    if not any((x for x in options if x in item)):
        raise F5ModuleError("A 'header_is_any' must be specified when the 'http_header' type is used.")
    if 'event' in item and item['event'] is not None:
        event = event_map.get(item['event'], None)
        if event:
            action[event] = True
    if 'header_is_any' in item:
        if isinstance(item['header_is_any'], list):
            values = item['header_is_any']
        else:
            values = [item['header_is_any']]
        action.update(dict(equals=True, tmName=item['header_name'], values=values))