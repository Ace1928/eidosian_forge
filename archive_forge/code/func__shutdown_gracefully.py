import sys
import struct
import ssl
from base64 import b64encode
from hashlib import sha1
import logging
from socket import error as SocketError
import errno
import threading
from socketserver import ThreadingMixIn, TCPServer, StreamRequestHandler
from websocket_server.thread import WebsocketServerThread
def _shutdown_gracefully(self, status=CLOSE_STATUS_NORMAL, reason=DEFAULT_CLOSE_REASON):
    """
        Send a CLOSE handshake to all connected clients before terminating server
        """
    self.keep_alive = False
    self._disconnect_clients_gracefully(status, reason)
    self.server_close()
    self.shutdown()