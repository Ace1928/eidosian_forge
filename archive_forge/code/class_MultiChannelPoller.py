from __future__ import annotations
import functools
import numbers
import socket
from bisect import bisect
from collections import namedtuple
from contextlib import contextmanager
from queue import Empty
from time import time
from vine import promise
from kombu.exceptions import InconsistencyError, VersionMismatch
from kombu.log import get_logger
from kombu.utils.compat import register_after_fork
from kombu.utils.encoding import bytes_to_str
from kombu.utils.eventio import ERR, READ, poll
from kombu.utils.functional import accepts_argument
from kombu.utils.json import dumps, loads
from kombu.utils.objects import cached_property
from kombu.utils.scheduling import cycle_by_name
from kombu.utils.url import _parse_url
from . import virtual
class MultiChannelPoller:
    """Async I/O poller for Redis transport."""
    eventflags = READ | ERR
    _in_protected_read = False
    after_read = None

    def __init__(self):
        self._channels = set()
        self._fd_to_chan = {}
        self._chan_to_sock = {}
        self.poller = poll()
        self.after_read = set()

    def close(self):
        for fd in self._chan_to_sock.values():
            try:
                self.poller.unregister(fd)
            except (KeyError, ValueError):
                pass
        self._channels.clear()
        self._fd_to_chan.clear()
        self._chan_to_sock.clear()

    def add(self, channel):
        self._channels.add(channel)

    def discard(self, channel):
        self._channels.discard(channel)

    def _on_connection_disconnect(self, connection):
        try:
            self.poller.unregister(connection._sock)
        except (AttributeError, TypeError):
            pass

    def _register(self, channel, client, type):
        if (channel, client, type) in self._chan_to_sock:
            self._unregister(channel, client, type)
        if client.connection._sock is None:
            client.connection.connect()
        sock = client.connection._sock
        self._fd_to_chan[sock.fileno()] = (channel, type)
        self._chan_to_sock[channel, client, type] = sock
        self.poller.register(sock, self.eventflags)

    def _unregister(self, channel, client, type):
        self.poller.unregister(self._chan_to_sock[channel, client, type])

    def _client_registered(self, channel, client, cmd):
        if getattr(client, 'connection', None) is None:
            client.connection = client.connection_pool.get_connection('_')
        return client.connection._sock is not None and (channel, client, cmd) in self._chan_to_sock

    def _register_BRPOP(self, channel):
        """Enable BRPOP mode for channel."""
        ident = (channel, channel.client, 'BRPOP')
        if not self._client_registered(channel, channel.client, 'BRPOP'):
            channel._in_poll = False
            self._register(*ident)
        if not channel._in_poll:
            channel._brpop_start()

    def _register_LISTEN(self, channel):
        """Enable LISTEN mode for channel."""
        if not self._client_registered(channel, channel.subclient, 'LISTEN'):
            channel._in_listen = False
            self._register(channel, channel.subclient, 'LISTEN')
        if not channel._in_listen:
            channel._subscribe()

    def on_poll_start(self):
        for channel in self._channels:
            if channel.active_queues:
                if channel.qos.can_consume():
                    self._register_BRPOP(channel)
            if channel.active_fanout_queues:
                self._register_LISTEN(channel)

    def on_poll_init(self, poller):
        self.poller = poller
        for channel in self._channels:
            return channel.qos.restore_visible(num=channel.unacked_restore_limit)

    def maybe_restore_messages(self):
        for channel in self._channels:
            if channel.active_queues:
                return channel.qos.restore_visible(num=channel.unacked_restore_limit)

    def maybe_check_subclient_health(self):
        for channel in self._channels:
            client = channel.__dict__.get('subclient')
            if client is not None and callable(getattr(client, 'check_health', None)):
                client.check_health()

    def on_readable(self, fileno):
        chan, type = self._fd_to_chan[fileno]
        if chan.qos.can_consume():
            chan.handlers[type]()

    def handle_event(self, fileno, event):
        if event & READ:
            return (self.on_readable(fileno), self)
        elif event & ERR:
            chan, type = self._fd_to_chan[fileno]
            chan._poll_error(type)

    def get(self, callback, timeout=None):
        self._in_protected_read = True
        try:
            for channel in self._channels:
                if channel.active_queues:
                    if channel.qos.can_consume():
                        self._register_BRPOP(channel)
                if channel.active_fanout_queues:
                    self._register_LISTEN(channel)
            events = self.poller.poll(timeout)
            if events:
                for fileno, event in events:
                    ret = self.handle_event(fileno, event)
                    if ret:
                        return
            self.maybe_restore_messages()
            raise Empty()
        finally:
            self._in_protected_read = False
            while self.after_read:
                try:
                    fun = self.after_read.pop()
                except KeyError:
                    break
                else:
                    fun()

    @property
    def fds(self):
        return self._fd_to_chan