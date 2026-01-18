from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import logging
import threading
import time
from googlecloudsdk.api_lib.compute import iap_tunnel_websocket_helper as helper
from googlecloudsdk.api_lib.compute import iap_tunnel_websocket_utils as utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import retry
import six
from six.moves import queue
def _StartNewWebSocket(self):
    """Start a new WebSocket and thread to listen for incoming data."""
    headers = ['User-Agent: ' + self._user_agent]
    log.debug('[%d] user-agent [%s]', self._conn_id, self._user_agent)
    request_reason = properties.VALUES.core.request_reason.Get()
    if request_reason:
        headers += ['X-Goog-Request-Reason: ' + request_reason]
    if self._get_access_token_callback:
        headers += ['Authorization: Bearer ' + self._get_access_token_callback()]
    log.debug('[%d] Using new websocket library', self._conn_id)
    if self._connection_sid:
        url = utils.CreateWebSocketReconnectUrl(self._tunnel_target, self._connection_sid, self._total_bytes_received, should_use_new_websocket=True)
        log.info('[%d] Reconnecting with URL [%r]', self._conn_id, url)
    else:
        url = utils.CreateWebSocketConnectUrl(self._tunnel_target, should_use_new_websocket=True)
        log.info('[%d] Connecting with URL [%r]', self._conn_id, url)
    self._connect_msg_received = False
    self._websocket_helper = helper.IapTunnelWebSocketHelper(url, headers, self._ignore_certs, self._tunnel_target.proxy_info, self._OnData, self._OnClose, should_use_new_websocket=True, conn_id=self._conn_id)
    self._websocket_helper.StartReceivingThread()