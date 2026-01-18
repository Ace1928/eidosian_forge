import os
import os.path
import socket
import ssl
import unittest
import websocket
import websocket as ws
from websocket._http import (
class OptsList:

    def __init__(self):
        self.timeout = 1
        self.sockopt = []
        self.sslopt = {'cert_reqs': ssl.CERT_NONE}