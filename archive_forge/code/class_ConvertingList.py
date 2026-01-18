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
class ConvertingList(list, ConvertingMixin):
    """A converting list wrapper."""

    def __getitem__(self, key):
        value = list.__getitem__(self, key)
        return self.convert_with_key(key, value)

    def pop(self, idx=-1):
        value = list.pop(self, idx)
        return self.convert(value)