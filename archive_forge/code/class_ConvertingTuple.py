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
class ConvertingTuple(tuple, ConvertingMixin):
    """A converting tuple wrapper."""

    def __getitem__(self, key):
        value = tuple.__getitem__(self, key)
        return self.convert_with_key(key, value, replace=False)