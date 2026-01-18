import errno
import io
import logging
import logging.handlers
import os
import queue
import re
import struct
import threading
import traceback
from socketserver import ThreadingTCPServer, StreamRequestHandler
class ConfigSocketReceiver(ThreadingTCPServer):
    """
        A simple TCP socket-based logging config receiver.
        """
    allow_reuse_address = 1

    def __init__(self, host='localhost', port=DEFAULT_LOGGING_CONFIG_PORT, handler=None, ready=None, verify=None):
        ThreadingTCPServer.__init__(self, (host, port), handler)
        logging._acquireLock()
        self.abort = 0
        logging._releaseLock()
        self.timeout = 1
        self.ready = ready
        self.verify = verify

    def serve_until_stopped(self):
        import select
        abort = 0
        while not abort:
            rd, wr, ex = select.select([self.socket.fileno()], [], [], self.timeout)
            if rd:
                self.handle_request()
            logging._acquireLock()
            abort = self.abort
            logging._releaseLock()
        self.server_close()