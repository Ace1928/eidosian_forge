import os
import io
import re
import email.utils
import socket
import sys
import time
import traceback as traceback_
import logging
import platform
import queue
import contextlib
import threading
import urllib.parse
from functools import lru_cache
from . import connections, errors, __version__
from ._compat import bton
from ._compat import IS_PPC
from .workers import threadpool
from .makefile import MakeFile, StreamWriter
class HeaderReader:
    """Object for reading headers from an HTTP request.

    Interface and default implementation.
    """

    def __call__(self, rfile, hdict=None):
        """
        Read headers from the given stream into the given header dict.

        If hdict is None, a new header dict is created. Returns the populated
        header dict.

        Headers which are repeated are folded together using a comma if their
        specification so dictates.

        This function raises ValueError when the read bytes violate the HTTP
        spec.
        You should probably return "400 Bad Request" if this happens.
        """
        if hdict is None:
            hdict = {}
        while True:
            line = rfile.readline()
            if not line:
                raise ValueError('Illegal end of headers.')
            if line == CRLF:
                break
            if not line.endswith(CRLF):
                raise ValueError('HTTP requires CRLF terminators')
            if line[0] in (SPACE, TAB):
                v = line.strip()
            else:
                try:
                    k, v = line.split(COLON, 1)
                except ValueError:
                    raise ValueError('Illegal header line.')
                v = v.strip()
                k = self._transform_key(k)
                hname = k
            if not self._allow_header(k):
                continue
            if k in comma_separated_headers:
                existing = hdict.get(hname)
                if existing:
                    v = b', '.join((existing, v))
            hdict[hname] = v
        return hdict

    def _allow_header(self, key_name):
        return True

    def _transform_key(self, key_name):
        return key_name.strip().title()