from xmlrpc.client import Fault, dumps, loads, gzip_encode, gzip_decode
from http.server import BaseHTTPRequestHandler
from functools import partial
from inspect import signature
import html
import http.server
import socketserver
import sys
import os
import re
import pydoc
import traceback
class SimpleXMLRPCServer(socketserver.TCPServer, SimpleXMLRPCDispatcher):
    """Simple XML-RPC server.

    Simple XML-RPC server that allows functions and a single instance
    to be installed to handle requests. The default implementation
    attempts to dispatch XML-RPC calls to the functions or instance
    installed in the server. Override the _dispatch method inherited
    from SimpleXMLRPCDispatcher to change this behavior.
    """
    allow_reuse_address = True
    _send_traceback_header = False

    def __init__(self, addr, requestHandler=SimpleXMLRPCRequestHandler, logRequests=True, allow_none=False, encoding=None, bind_and_activate=True, use_builtin_types=False):
        self.logRequests = logRequests
        SimpleXMLRPCDispatcher.__init__(self, allow_none, encoding, use_builtin_types)
        socketserver.TCPServer.__init__(self, addr, requestHandler, bind_and_activate)