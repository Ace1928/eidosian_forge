from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import compare_complex_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _handle_tcp_condition(self, action, item):
    options = ['address_matches_with_any', 'address_matches_with_datagroup', 'address_matches_with_external_datagroup']
    event_map = dict(client_accepted='clientAccepted', proxy_connect='proxyConnect', proxy_request='proxyRequest', proxy_response='proxyResponse', request='request', ssl_client_hello='sslClientHello', ssl_client_server_hello_send='sslClientServerhelloSend')
    action['type'] = 'tcp'
    if all((k not in item for k in options)):
        raise F5ModuleError("A 'address_matches_with_any','address_matches_with_datagroup' or'address_matches_with_external_datagroup' must be specified when the 'tcp' type is used.")
    if 'address_matches_with_any' in item and item['address_matches_with_any'] is not None:
        if isinstance(item['address_matches_with_any'], list):
            values = item['address_matches_with_any']
        else:
            values = [item['address_matches_with_any']]
        action.update(dict(address=True, matches=True, values=values))
    if 'address_matches_with_datagroup' in item and item['address_matches_with_datagroup'] is not None:
        if isinstance(item['address_matches_with_datagroup'], list):
            values = item['address_matches_with_datagroup']
        else:
            values = [item['address_matches_with_datagroup']]
        for value in values:
            action.update(dict(address=True, matches=True, datagroup=fq_name(self.partition, value)))
    if 'address_matches_with_external_datagroup' in item and item['address_matches_with_external_datagroup'] is not None:
        if isinstance(item['address_matches_with_external_datagroup'], list):
            values = item['address_matches_with_external_datagroup']
        else:
            values = [item['address_matches_with_external_datagroup']]
        for value in values:
            action.update(dict(address=True, matches=True, datagroup=fq_name(self.partition, value)))
    if 'event' in item and item['event'] is not None:
        event = event_map.get(item['event'], None)
        if event:
            action[event] = True