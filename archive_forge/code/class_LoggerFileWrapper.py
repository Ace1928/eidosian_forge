import errno
import os
import sys
import time
import traceback
import types
import urllib.parse
import warnings
import eventlet
from eventlet import greenio
from eventlet import support
from eventlet.corolocal import local
from eventlet.green import BaseHTTPServer
from eventlet.green import socket
class LoggerFileWrapper(LoggerNull):

    def __init__(self, log, debug):
        self.log = log
        self._debug = debug

    def error(self, msg, *args, **kwargs):
        self.write(msg, *args)

    def info(self, msg, *args, **kwargs):
        self.write(msg, *args)

    def debug(self, msg, *args, **kwargs):
        if self._debug:
            self.write(msg, *args)

    def write(self, msg, *args):
        msg = msg + '\n'
        if args:
            msg = msg % args
        self.log.write(msg)