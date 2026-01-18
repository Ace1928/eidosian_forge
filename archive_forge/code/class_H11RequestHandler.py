import json
import os.path
import socket
import socketserver
import threading
from contextlib import closing, contextmanager
from http.server import SimpleHTTPRequestHandler
from typing import Callable, Generator
from urllib.request import urlopen
import h11
class H11RequestHandler(socketserver.BaseRequestHandler):

    def handle(self) -> None:
        with closing(self.request) as s:
            c = h11.Connection(h11.SERVER)
            request = None
            while True:
                event = c.next_event()
                if event is h11.NEED_DATA:
                    c.receive_data(s.recv(10))
                    continue
                if type(event) is h11.Request:
                    request = event
                if type(event) is h11.EndOfMessage:
                    break
            assert request is not None
            info = json.dumps({'method': request.method.decode('ascii'), 'target': request.target.decode('ascii'), 'headers': {name.decode('ascii'): value.decode('ascii') for name, value in request.headers}})
            s.sendall(c.send(h11.Response(status_code=200, headers=[])))
            s.sendall(c.send(h11.Data(data=info.encode('ascii'))))
            s.sendall(c.send(h11.EndOfMessage()))