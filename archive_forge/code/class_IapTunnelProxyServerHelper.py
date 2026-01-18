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
class IapTunnelProxyServerHelper:
    """Proxy server helper listens on a port for new local connections."""

    def __init__(self, local_host, local_port, should_test_connection, tunneler):
        self._tunneler = tunneler
        self._local_host = local_host
        self._local_port = local_port
        self._should_test_connection = should_test_connection
        self._server_sockets = []
        self._connections = []
        self._total_connections = 0

    def __del__(self):
        self._CloseServerSockets()

    def Run(self):
        """Start accepting connections."""
        if self._should_test_connection:
            try:
                self._TestConnection()
            except iap_tunnel_websocket.ConnectionCreationError as e:
                raise iap_tunnel_websocket.ConnectionCreationError('While checking if a connection can be made: %s' % six.text_type(e))
        self._server_sockets = _OpenLocalTcpSockets(self._local_host, self._local_port)
        log.out.Print('Listening on port [%d].' % self._local_port)
        try:
            with execution_utils.RaisesKeyboardInterrupt():
                while True:
                    self._connections.append(self._AcceptNewConnection())
                    self._CleanDeadClientConnections()
        except KeyboardInterrupt:
            log.info('Keyboard interrupt received.')
        finally:
            self._CloseServerSockets()
        self._tunneler.Close()
        self._CloseClientConnections()
        log.status.Print('Server shutdown complete.')

    def _TestConnection(self):
        log.status.Print('Testing if tunnel connection works.')
        user_agent = transport.MakeUserAgentString()
        conn = self._tunneler._InitiateConnection(None, functools.partial(_GetAccessTokenCallback, store.LoadIfEnabled(use_google_auth=True)), user_agent)
        conn.Close()

    def _AcceptNewConnection(self):
        """Accept a new socket connection and start a new WebSocket tunnel."""
        ready_sockets = [()]
        while not ready_sockets[0]:
            ready_sockets = select.select(self._server_sockets, (), (), 0.2)
        ready_read_sockets = ready_sockets[0]
        conn, socket_address = ready_read_sockets[0].accept()
        new_thread = threading.Thread(target=self._HandleNewConnection, args=(conn, socket_address, self._total_connections))
        new_thread.daemon = True
        new_thread.start()
        self._total_connections += 1
        return (new_thread, conn)

    def _CloseServerSockets(self):
        log.debug('Stopping server.')
        try:
            for server_socket in self._server_sockets:
                server_socket.close()
        except EnvironmentError:
            pass

    def _CloseClientConnections(self):
        """Close client connections that seem to still be open."""
        if self._connections:
            close_count = 0
            for client_thread, conn in self._connections:
                if client_thread.is_alive():
                    close_count += 1
                    try:
                        conn.close()
                    except EnvironmentError:
                        pass
            if close_count:
                log.status.Print('Closed [%d] local connection(s).' % close_count)

    def _CleanDeadClientConnections(self):
        """Erase reference to dead connections so they can be garbage collected."""
        conn_still_alive = []
        if self._connections:
            dead_connections = 0
            for client_thread, conn in self._connections:
                if not client_thread.is_alive():
                    dead_connections += 1
                    try:
                        conn.close()
                    except EnvironmentError:
                        pass
                    del conn
                    del client_thread
                else:
                    conn_still_alive.append([client_thread, conn])
            if dead_connections:
                log.debug('Cleaned [%d] dead connection(s).' % dead_connections)
                self._connections = conn_still_alive
            gc.collect(2)
            log.debug('connections alive: [%d]' % len(self._connections))

    def _HandleNewConnection(self, conn, socket_address, conn_id):
        try:
            user_agent = transport.MakeUserAgentString()
            self._tunneler.RunReceiveLocalData(conn, repr(socket_address), user_agent, conn_id=conn_id)
        except EnvironmentError as e:
            log.info('Socket error [%s] while receiving from client.', six.text_type(e))
        except:
            log.exception('Error while receiving from client.')