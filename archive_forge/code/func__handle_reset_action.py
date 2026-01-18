from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import compare_complex_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _handle_reset_action(self, action, item):
    """Handle the nuances of the reset type

        :param action:
        :param item:
        :return:
        """
    event_map = dict(client_accepted='clientAccepted', proxy_connect='proxyConnect', proxy_request='proxyRequest', proxy_response='proxyResponse', request='request', response='response', server_connected='serverConnected', ssl_client_hello='sslClientHello', ssl_client_server_hello_send='sslClientServerhelloSend', ssl_server_handshake='sslServerHandshake', ssl_server_hello='sslServerHello', websocket_request='wsRequest', websocket_response='wsResponse')
    action['type'] = 'reset'
    if 'event' not in item:
        raise F5ModuleError("An 'event' must be specified when the 'reset' type is used.")
    event = event_map.get(item['event'], None)
    if not event:
        raise F5ModuleError('Invalid event type specified for reset action: {0},check module documentation for valid event types.'.format(item['event']))
    action[event] = True
    action.update({'connection': True, 'shutdown': True})