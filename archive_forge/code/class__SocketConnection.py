import collections
import errno
import heapq
import logging
import math
import os
import pyngus
import select
import socket
import threading
import time
import uuid
class _SocketConnection(object):
    """Associates a pyngus Connection with a python network socket,
    and handles all connection-related I/O and timer events.
    """

    def __init__(self, name, container, properties, handler):
        self.name = name
        self.socket = None
        self.pyngus_conn = None
        self._properties = properties
        self._handler = handler
        self._container = container

    def fileno(self):
        """Allows use of a _SocketConnection in a select() call.
        """
        return self.socket.fileno()

    def read_socket(self):
        """Called to read from the socket."""
        if self.socket:
            try:
                pyngus.read_socket_input(self.pyngus_conn, self.socket)
                self.pyngus_conn.process(time.monotonic())
            except (socket.timeout, socket.error) as e:
                self.pyngus_conn.close_input()
                self.pyngus_conn.close_output()
                self._handler.socket_error(str(e))

    def write_socket(self):
        """Called to write to the socket."""
        if self.socket:
            try:
                pyngus.write_socket_output(self.pyngus_conn, self.socket)
                self.pyngus_conn.process(time.monotonic())
            except (socket.timeout, socket.error) as e:
                self.pyngus_conn.close_output()
                self.pyngus_conn.close_input()
                self._handler.socket_error(str(e))

    def connect(self, host):
        """Connect to host and start the AMQP protocol."""
        addr = socket.getaddrinfo(host.hostname, host.port, socket.AF_UNSPEC, socket.SOCK_STREAM)
        if not addr:
            key = '%s:%i' % (host.hostname, host.port)
            error = "Invalid peer address '%s'" % key
            LOG.error("Invalid peer address '%s'", key)
            self._handler.socket_error(error)
            return
        my_socket = socket.socket(addr[0][0], addr[0][1], addr[0][2])
        my_socket.setblocking(0)
        my_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            my_socket.connect(addr[0][4])
        except socket.error as e:
            if e.errno != errno.EINPROGRESS:
                error = "Socket connect failure '%s'" % str(e)
                LOG.error("Socket connect failure '%s'", str(e))
                self._handler.socket_error(error)
                return
        self.socket = my_socket
        props = self._properties.copy()
        if pyngus.VERSION >= (2, 0, 0):
            props['x-server'] = False
            if host.username:
                props['x-username'] = host.username
                props['x-password'] = host.password or ''
        self.pyngus_conn = self._container.create_connection(self.name, self._handler, props)
        self.pyngus_conn.user_context = self
        if pyngus.VERSION < (2, 0, 0):
            pn_sasl = self.pyngus_conn.pn_sasl
            if host.username:
                password = host.password if host.password else ''
                pn_sasl.plain(host.username, password)
            else:
                pn_sasl.mechanisms('ANONYMOUS')
                pn_sasl.client()
        self.pyngus_conn.open()

    def reset(self, name=None):
        """Clean up the current state, expect 'connect()' to be recalled
        later.
        """
        if self.pyngus_conn:
            self.pyngus_conn.destroy()
            self.pyngus_conn = None
        self.close()
        if name:
            self.name = name

    def close(self):
        if self.socket:
            self.socket.close()
            self.socket = None