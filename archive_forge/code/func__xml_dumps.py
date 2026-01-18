import io
import os
import sys
import socket
import struct
import time
import tempfile
import itertools
from . import util
from . import AuthenticationError, BufferTooShort
from .context import reduction
def _xml_dumps(obj):
    return xmlrpclib.dumps((obj,), None, None, None, 1).encode('utf-8')