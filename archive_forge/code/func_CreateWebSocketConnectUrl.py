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
def CreateWebSocketConnectUrl(tunnel_target, should_use_new_websocket):
    """Create Connect URL for WebSocket connection."""
    url_query_pieces = {'project': tunnel_target.project, 'port': tunnel_target.port, 'newWebsocket': should_use_new_websocket}
    if tunnel_target.host:
        url_query_pieces['region'] = tunnel_target.region
        url_query_pieces['network'] = tunnel_target.network
        url_query_pieces['host'] = tunnel_target.host
        if tunnel_target.dest_group:
            url_query_pieces['group'] = tunnel_target.dest_group
    else:
        url_query_pieces['zone'] = tunnel_target.zone
        url_query_pieces['instance'] = tunnel_target.instance
        url_query_pieces['interface'] = tunnel_target.interface
    return _CreateWebSocketUrl(CONNECT_ENDPOINT, url_query_pieces, tunnel_target.url_override)