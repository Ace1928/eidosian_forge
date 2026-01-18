from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import compare_complex_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _handle_http_method_condition(self, action, item):
    options = ['method_matches_with_any']
    action['type'] = 'http_method'
    event_map = dict(proxy_connect='proxyConnect', proxy_request='proxyRequest', proxy_response='proxyResponse', request='request', response='response')
    if not any((x for x in options if x in item)):
        raise F5ModuleError("A 'method_matches_with_any' must be specified when the 'http_method' type is used.")
    if 'event' in item and item['event'] is not None:
        event = event_map.get(item['event'], None)
        if event:
            action[event] = True
    if 'method_matches_with_any' in item and item['method_matches_with_any'] is not None:
        if isinstance(item['method_matches_with_any'], list):
            values = item['method_matches_with_any']
        else:
            values = [item['method_matches_with_any']]
        action.update(dict(startsWith=True, values=values))