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
def _clearExistingHandlers():
    """Clear and close existing handlers"""
    logging._handlers.clear()
    logging.shutdown(logging._handlerList[:])
    del logging._handlerList[:]