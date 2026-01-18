import os
import select
import signal
import socket
import sys
import threading
from wsgiref.simple_server import make_server
from oslo_config import cfg
from oslo_log import log as logging
from prometheus_client import make_wsgi_app
from oslo_metrics import message_router
class MetricsListener:

    def __init__(self, socket_path):
        self.socket_path = socket_path
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.unlink(socket_path)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.socket_path)
        self.start = True
        self.router = message_router.MessageRouter()

    def unlink(self, socket_path):
        try:
            os.unlink(socket_path)
        except OSError:
            if os.path.exists(socket_path):
                raise

    def serve(self):
        while self.start:
            readable, writable, exceptional = select.select([self.socket], [], [], 1)
            if len(readable) == 0:
                continue
            try:
                LOG.debug('wait for socket.recv')
                msg = self.socket.recv(65565)
                LOG.debug('got message')
                self.router.process(msg)
            except socket.timeout:
                pass

    def stop(self):
        self.socket.close()
        self.start = False