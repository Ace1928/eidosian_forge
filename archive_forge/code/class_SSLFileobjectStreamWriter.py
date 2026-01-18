import socket
import sys
import threading
import time
from . import Adapter
from .. import errors, server as cheroot_server
from ..makefile import StreamReader, StreamWriter
class SSLFileobjectStreamWriter(SSLFileobjectMixin, StreamWriter):
    """SSL file object attached to a socket object."""