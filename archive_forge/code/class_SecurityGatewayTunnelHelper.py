from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ctypes
import errno
import functools
import gc
import io
import os
import select
import socket
import sys
import threading
from googlecloudsdk.api_lib.compute import iap_tunnel_websocket
from googlecloudsdk.api_lib.compute import iap_tunnel_websocket_utils as utils
from googlecloudsdk.api_lib.compute import sg_tunnel
from googlecloudsdk.api_lib.compute import sg_tunnel_utils as sg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import http_proxy
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.credentials import creds
from googlecloudsdk.core.credentials import store
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import portpicker
import six
from six.moves import queue
class SecurityGatewayTunnelHelper(object):
    """Helper class for starting a Security Gateaway tunnel."""

    def __init__(self, args, project, region, security_gateway, host, port):
        self._tunnel_url_override = args.iap_tunnel_url_override
        self._ignore_certs = args.iap_tunnel_insecure_disable_websocket_cert_check
        self._project = project
        self._region = region
        self._security_gateway = security_gateway
        self._host = host
        self._port = port
        self._shutdown = False

    def _InitiateConnection(self, local_conn, get_access_token_callback, user_agent):
        del user_agent
        sg_tunnel_target = self._GetTargetInfo()
        new_sg_tunnel = sg_tunnel.SecurityGatewayTunnel(sg_tunnel_target, get_access_token_callback, functools.partial(_SendLocalDataCallback, local_conn), functools.partial(_CloseLocalConnectionCallback, local_conn), self._ignore_certs)
        new_sg_tunnel.InitiateConnection()
        return new_sg_tunnel

    def _GetTargetInfo(self):
        proxy_info = http_proxy.GetHttpProxyInfo()
        if callable(proxy_info):
            proxy_info = proxy_info(method='https')
        return sg_utils.SecurityGatewayTargetInfo(project=self._project, region=self._region, security_gateway=self._security_gateway, host=self._host, port=self._port, url_override=self._tunnel_url_override, proxy_info=proxy_info)

    def RunReceiveLocalData(self, local_conn, socket_address, user_agent, conn_id=-1):
        """Receive data from provided local connection and send over HTTP CONNECT.

    Args:
      local_conn: A socket or _StdinSocket representing the local connection.
      socket_address: A verbose loggable string describing where conn is
        connected to.
      user_agent: The user_agent of this connection
      conn_id: The id of the connection.
    """
        del conn_id
        sg_conn = None
        try:
            sg_conn = self._InitiateConnection(local_conn, functools.partial(_GetAccessTokenCallback, store.LoadIfEnabled(use_google_auth=True)), user_agent)
            while not (self._shutdown or sg_conn.ShouldStop()):
                data = local_conn.recv(utils.SUBPROTOCOL_MAX_DATA_FRAME_SIZE)
                if not data:
                    log.warning('Local connection [%s] has closed.', socket_address)
                    break
                sg_conn.Send(data)
        except socket.error as e:
            log.error('Error while transmitting local connection [%s]: %s ', socket_address, e)
        finally:
            log.info('Terminating connection from local connection: [%s]', socket_address)
            if local_conn:
                local_conn.shutdown(socket.SHUT_RD)
                local_conn.close()
            if sg_conn:
                sg_conn.Close()
                log.debug('Connection [%s] closed.', socket_address)

    def Close(self):
        self._shutdown = True