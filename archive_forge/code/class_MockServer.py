import sys
import threading
import webbrowser
import socket
from http import server
from io import BytesIO as IO
import itertools
import random
class MockServer:

    def __init__(self, ip_port, Handler):
        Handler(MockRequest(), ip_port[0], self)

    def serve_forever(self):
        pass

    def server_close(self):
        pass