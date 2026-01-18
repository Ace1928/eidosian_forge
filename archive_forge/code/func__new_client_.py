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
def _new_client_(self, handler):
    if self._deny_clients:
        status = self._deny_clients['status']
        reason = self._deny_clients['reason']
        handler.send_close(status, reason)
        self._terminate_client_handler(handler)
        return
    self.id_counter += 1
    client = {'id': self.id_counter, 'handler': handler, 'address': handler.client_address}
    self.clients.append(client)
    self.new_client(client, self)