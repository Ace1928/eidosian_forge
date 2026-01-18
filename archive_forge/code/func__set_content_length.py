from __future__ import (absolute_import, division,
from future.builtins import bytes, int, str, super
from future.utils import PY2
from future.backports.email import parser as email_parser
from future.backports.email import message as email_message
from future.backports.misc import create_connection as socket_create_connection
import io
import os
import socket
from future.backports.urllib.parse import urlsplit
import warnings
from array import array
def _set_content_length(self, body):
    thelen = None
    try:
        thelen = str(len(body))
    except TypeError as te:
        try:
            thelen = str(os.fstat(body.fileno()).st_size)
        except (AttributeError, OSError):
            if self.debuglevel > 0:
                print('Cannot stat!!')
    if thelen is not None:
        self.putheader('Content-Length', thelen)