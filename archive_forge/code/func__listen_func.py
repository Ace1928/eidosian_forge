import errno
import gc
import logging
import os
import pprint
import sys
import tempfile
import traceback
import eventlet.backdoor
import greenlet
import yappi
from eventlet.green import socket
from oslo_service._i18n import _
from oslo_service import _options
def _listen_func(host, port):
    try:
        return eventlet.listen((host, port), reuse_port=False)
    except TypeError:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((host, port))
        sock.listen(50)
        return sock