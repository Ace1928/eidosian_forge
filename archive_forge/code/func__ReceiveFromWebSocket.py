from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import logging
import ssl
import sys
import threading
import traceback
from googlecloudsdk.api_lib.compute import iap_tunnel_lightweight_websocket as iap_websocket
from googlecloudsdk.api_lib.compute import iap_tunnel_websocket_utils as utils
from googlecloudsdk.core import context_aware
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import encoding
import six
import websocket
def _ReceiveFromWebSocket(self):
    """Receive data from WebSocket connection."""
    try:
        if self._proxy_info:
            http_proxy_auth = None
            if self._proxy_info.proxy_user or self._proxy_info.proxy_pass:
                http_proxy_auth = (encoding.Decode(self._proxy_info.proxy_user), encoding.Decode(self._proxy_info.proxy_pass))
            self._websocket.run_forever(origin=TUNNEL_CLOUDPROXY_ORIGIN, sslopt=self._sslopt, http_proxy_host=self._proxy_info.proxy_host, http_proxy_port=self._proxy_info.proxy_port, http_proxy_auth=http_proxy_auth)
        else:
            self._websocket.run_forever(origin=TUNNEL_CLOUDPROXY_ORIGIN, sslopt=self._sslopt)
    except:
        try:
            log.info('[%d] Error while receiving from WebSocket.', self._conn_id, exc_info=True)
        except:
            pass
    try:
        self.Close()
    except:
        try:
            log.info('[%d] Error while closing in receiving thread.', self._conn_id, exc_info=True)
        except:
            pass