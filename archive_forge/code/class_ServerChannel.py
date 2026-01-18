import datetime
import enum
import logging
import socket
import sys
import threading
import msgpack
from oslo_privsep._i18n import _
from oslo_utils import uuidutils
class ServerChannel(object):
    """Server-side twin to ClientChannel"""

    def __init__(self, sock):
        self.rlock = threading.Lock()
        self.reader_iter = iter(Deserializer(sock))
        self.wlock = threading.Lock()
        self.writer = Serializer(sock)

    def __iter__(self):
        return self

    def __next__(self):
        with self.rlock:
            return next(self.reader_iter)

    def send(self, msg):
        with self.wlock:
            self.writer.send(msg)