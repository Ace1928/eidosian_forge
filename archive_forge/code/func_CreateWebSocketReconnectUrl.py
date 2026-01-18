from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
import struct
import sys
from googlecloudsdk.core import context_aware
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import http_proxy_types
import httplib2
import six
from six.moves.urllib import parse
import socks
def CreateWebSocketReconnectUrl(tunnel_target, sid, ack_bytes, should_use_new_websocket):
    """Create Reconnect URL for WebSocket connection."""
    url_query_pieces = {'sid': sid, 'ack': ack_bytes, 'newWebsocket': should_use_new_websocket}
    if tunnel_target.host:
        url_query_pieces['region'] = tunnel_target.region
    else:
        url_query_pieces['zone'] = tunnel_target.zone
    return _CreateWebSocketUrl(RECONNECT_ENDPOINT, url_query_pieces, tunnel_target.url_override)