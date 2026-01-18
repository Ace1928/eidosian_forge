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
def _terminate_client_handler(self, handler):
    handler.keep_alive = False
    handler.finish()
    handler.connection.close()