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
def add_handlers(self, logger, handlers):
    """Add handlers to a logger from a list of names."""
    for h in handlers:
        try:
            logger.addHandler(self.config['handlers'][h])
        except Exception as e:
            raise ValueError('Unable to add handler %r' % h) from e