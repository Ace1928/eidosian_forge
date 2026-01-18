from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import compare_complex_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _handle_forward_action(self, action, item):
    """Handle the nuances of the forwarding type

        Right now there is only a single type of forwarding that can be done. As that
        functionality expands, so-to will the behavior of this, and other, methods.
        Therefore, do not be surprised that the logic here is so rigid. It's deliberate.

        :param action:
        :param item:
        :return:
        """
    event_map = dict(client_accepted='clientAccepted', proxy_request='proxyRequest', request='request', ssl_client_hello='sslClientHello', ssl_client_server_hello_send='sslClientServerhelloSend')
    action['type'] = 'forward'
    options = ['pool', 'virtual', 'node']
    if not any((x for x in options if x in item)):
        raise F5ModuleError("A 'pool' or 'virtual' or 'node' must be specified when the 'forward' type is used.")
    if item.get('pool', None):
        action['pool'] = fq_name(self.partition, item['pool'])
    elif item.get('virtual', None):
        action['virtual'] = fq_name(self.partition, item['virtual'])
    elif item.get('node', None):
        action['node'] = item['node']
    if 'event' in item and item['event'] is not None:
        event = event_map.get(item['event'], None)
        if event:
            action[event] = True