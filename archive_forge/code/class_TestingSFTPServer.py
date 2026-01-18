import os
import socket
import socketserver
import sys
import time
import paramiko
from .. import osutils, trace, urlutils
from ..transport import ssh
from . import test_server
class TestingSFTPServer(test_server.TestingThreadingTCPServer):

    def __init__(self, server_address, request_handler_class, test_case_server):
        test_server.TestingThreadingTCPServer.__init__(self, server_address, request_handler_class)
        self.test_case_server = test_case_server