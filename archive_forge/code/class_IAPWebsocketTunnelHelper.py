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
class IAPWebsocketTunnelHelper(object):
    """Helper class for starting an IAP WebSocket tunnel."""

    def __init__(self, args, project, zone=None, instance=None, interface=None, port=None, region=None, network=None, host=None, dest_group=None):
        self._project = project
        self._iap_tunnel_url_override = args.iap_tunnel_url_override
        self._ignore_certs = args.iap_tunnel_insecure_disable_websocket_cert_check
        self._zone = zone
        self._instance = instance
        self._interface = interface
        self._port = port
        self._region = region
        self._network = network
        self._host = host
        self._dest_group = dest_group
        self._shutdown = False

    def Close(self):
        self._shutdown = True

    def _InitiateConnection(self, local_conn, get_access_token_callback, user_agent, conn_id=-1):
        tunnel_target = self._GetTunnelTargetInfo()
        new_websocket = iap_tunnel_websocket.IapTunnelWebSocket(tunnel_target, get_access_token_callback, functools.partial(_SendLocalDataCallback, local_conn), functools.partial(_CloseLocalConnectionCallback, local_conn), user_agent, ignore_certs=self._ignore_certs, conn_id=conn_id)
        new_websocket.InitiateConnection()
        return new_websocket

    def _GetTunnelTargetInfo(self):
        proxy_info = http_proxy.GetHttpProxyInfo()
        if callable(proxy_info):
            proxy_info = proxy_info(method='https')
        return utils.IapTunnelTargetInfo(project=self._project, zone=self._zone, instance=self._instance, interface=self._interface, port=self._port, url_override=self._iap_tunnel_url_override, proxy_info=proxy_info, region=self._region, network=self._network, host=self._host, dest_group=self._dest_group)

    def RunReceiveLocalData(self, conn, socket_address, user_agent, conn_id=0):
        """Receive data from provided local connection and send over WebSocket.

    Args:
      conn: A socket or _StdinSocket representing the local connection.
      socket_address: A verbose loggable string describing where conn is
        connected to.
      user_agent: The user_agent of this connection
      conn_id: Id of the connection.
    """
        websocket_conn = None
        try:
            websocket_conn = self._InitiateConnection(conn, functools.partial(_GetAccessTokenCallback, store.LoadIfEnabled(use_google_auth=True)), user_agent, conn_id=conn_id)
            while not self._shutdown:
                data = conn.recv(utils.SUBPROTOCOL_MAX_DATA_FRAME_SIZE)
                if not data:
                    websocket_conn.LocalEOF()
                    if not websocket_conn.WaitForAllSent():
                        log.warning('[%d] Failed to send all data from [%s].', conn_id, socket_address)
                    break
                websocket_conn.Send(data)
        finally:
            if self._shutdown:
                log.info('[%d] Terminating connection to [%s].', conn_id, socket_address)
            else:
                log.info('[%d] Client closed connection from [%s].', conn_id, socket_address)
            try:
                conn.close()
            except EnvironmentError:
                pass
            try:
                if websocket_conn:
                    websocket_conn.Close()
            except (EnvironmentError, exceptions.Error):
                pass