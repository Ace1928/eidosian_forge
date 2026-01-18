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
def _run_forever(self, threaded):
    cls_name = self.__class__.__name__
    try:
        logger.info('Listening on port %d for clients..' % self.port)
        if threaded:
            self.daemon = True
            self.thread = WebsocketServerThread(target=super().serve_forever, daemon=True, logger=logger)
            logger.info(f'Starting {cls_name} on thread {self.thread.getName()}.')
            self.thread.start()
        else:
            self.thread = threading.current_thread()
            logger.info(f'Starting {cls_name} on main thread.')
            super().serve_forever()
    except KeyboardInterrupt:
        self.server_close()
        logger.info('Server terminated.')
    except Exception as e:
        logger.error(str(e), exc_info=True)
        sys.exit(1)