import array
from contextlib import contextmanager
import errno
from itertools import count
import logging
from typing import Optional
from outcome import Value, Error
import trio
from trio.abc import Channel
from jeepney.auth import Authenticator, BEGIN
from jeepney.bus import get_bus
from jeepney.fds import FileDescriptor, fds_buf_size
from jeepney.low_level import Parser, MessageType, Message
from jeepney.wrappers import ProxyBase, unwrap_msg
from jeepney.bus_messages import message_bus
from .common import (
class DBusRouter:
    """A client D-Bus connection which can wait for replies.

    This runs a separate receiver task and dispatches received messages.
    """
    _nursery_mgr = None
    _rcv_cancel_scope = None

    def __init__(self, conn: DBusConnection):
        self._conn = conn
        self._replies = ReplyMatcher()
        self._filters = MessageFilters()

    @property
    def unique_name(self):
        return self._conn.unique_name

    async def send(self, message, *, serial=None):
        """Send a message, don't wait for a reply
        """
        await self._conn.send(message, serial=serial)

    async def send_and_get_reply(self, message) -> Message:
        """Send a method call message and wait for the reply

        Returns the reply message (method return or error message type).
        """
        check_replyable(message)
        if self._rcv_cancel_scope is None:
            raise RouterClosed('This DBusRouter has stopped')
        serial = next(self._conn.outgoing_serial)
        with self._replies.catch(serial, Future()) as reply_fut:
            await self.send(message, serial=serial)
            return await reply_fut.get()

    def filter(self, rule, *, channel: Optional[trio.MemorySendChannel]=None, bufsize=1):
        """Create a filter for incoming messages

        Usage::

            async with router.filter(rule) as receive_channel:
                matching_msg = await receive_channel.receive()

            # OR:
            send_chan, recv_chan = trio.open_memory_channel(1)
            async with router.filter(rule, channel=send_chan):
                matching_msg = await recv_chan.receive()

        If the channel fills up,
        The sending end of the channel is closed when leaving the ``async with``
        block, whether or not it was passed in.

        :param jeepney.MatchRule rule: Catch messages matching this rule
        :param trio.MemorySendChannel channel: Send matching messages here
        :param int bufsize: If no channel is passed in, create one with this size
        """
        if channel is None:
            channel, recv_channel = trio.open_memory_channel(bufsize)
        else:
            recv_channel = None
        return TrioFilterHandle(self._filters, rule, channel, recv_channel)

    async def start(self, nursery: trio.Nursery):
        if self._rcv_cancel_scope is not None:
            raise RuntimeError('DBusRouter receiver task is already running')
        self._rcv_cancel_scope = await nursery.start(self._receiver)

    async def aclose(self):
        """Stop the sender & receiver tasks"""
        if self._rcv_cancel_scope is not None:
            self._rcv_cancel_scope.cancel()
            self._rcv_cancel_scope = None
        await trio.sleep(0)

    def _dispatch(self, msg: Message):
        """Handle one received message"""
        if self._replies.dispatch(msg):
            return
        for filter in self._filters.matches(msg):
            try:
                filter.send_channel.send_nowait(msg)
            except trio.WouldBlock:
                pass

    async def _receiver(self, task_status=trio.TASK_STATUS_IGNORED):
        """Receiver loop - runs in a separate task"""
        with trio.CancelScope() as cscope:
            self.is_running = True
            task_status.started(cscope)
            try:
                while True:
                    msg = await self._conn.receive()
                    self._dispatch(msg)
            finally:
                self.is_running = False
                self._replies.drop_all()
                with trio.move_on_after(3) as cleanup_scope:
                    for filter in self._filters.filters.values():
                        cleanup_scope.shield = True
                        await filter.send_channel.aclose()