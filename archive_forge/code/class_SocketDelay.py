import os
import socket
import socketserver
import sys
import time
import paramiko
from .. import osutils, trace, urlutils
from ..transport import ssh
from . import test_server
class SocketDelay:
    """A socket decorator to make TCP appear slower.

    This changes recv, send, and sendall to add a fixed latency to each python
    call if a new roundtrip is detected. That is, when a recv is called and the
    flag new_roundtrip is set, latency is charged. Every send and send_all
    sets this flag.

    In addition every send, sendall and recv sleeps a bit per character send to
    simulate bandwidth.

    Not all methods are implemented, this is deliberate as this class is not a
    replacement for the builtin sockets layer. fileno is not implemented to
    prevent the proxy being bypassed.
    """
    simulated_time = 0
    _proxied_arguments = dict.fromkeys(['close', 'getpeername', 'getsockname', 'getsockopt', 'gettimeout', 'setblocking', 'setsockopt', 'settimeout', 'shutdown'])

    def __init__(self, sock, latency, bandwidth=1.0, really_sleep=True):
        """
        :param bandwith: simulated bandwith (MegaBit)
        :param really_sleep: If set to false, the SocketDelay will just
        increase a counter, instead of calling time.sleep. This is useful for
        unittesting the SocketDelay.
        """
        self.sock = sock
        self.latency = latency
        self.really_sleep = really_sleep
        self.time_per_byte = 1 / (bandwidth / 8.0 * 1024 * 1024)
        self.new_roundtrip = False

    def sleep(self, s):
        if self.really_sleep:
            time.sleep(s)
        else:
            SocketDelay.simulated_time += s

    def __getattr__(self, attr):
        if attr in SocketDelay._proxied_arguments:
            return getattr(self.sock, attr)
        raise AttributeError("'SocketDelay' object has no attribute %r" % attr)

    def dup(self):
        return SocketDelay(self.sock.dup(), self.latency, self.time_per_byte, self._sleep)

    def recv(self, *args):
        data = self.sock.recv(*args)
        if data and self.new_roundtrip:
            self.new_roundtrip = False
            self.sleep(self.latency)
        self.sleep(len(data) * self.time_per_byte)
        return data

    def sendall(self, data, flags=0):
        if not self.new_roundtrip:
            self.new_roundtrip = True
            self.sleep(self.latency)
        self.sleep(len(data) * self.time_per_byte)
        return self.sock.sendall(data, flags)

    def send(self, data, flags=0):
        if not self.new_roundtrip:
            self.new_roundtrip = True
            self.sleep(self.latency)
        bytes_sent = self.sock.send(data, flags)
        self.sleep(bytes_sent * self.time_per_byte)
        return bytes_sent