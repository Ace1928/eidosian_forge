from __future__ import annotations
import socket
from collections import deque
from queue import Empty
from time import monotonic
from typing import TYPE_CHECKING
from . import entity, messaging
from .connection import maybe_channel
class SimpleBase:
    Empty = Empty
    _consuming = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> None:
        self.close()

    def __init__(self, channel, producer, consumer, no_ack=False):
        self.channel = maybe_channel(channel)
        self.producer = producer
        self.consumer = consumer
        self.no_ack = no_ack
        self.queue = self.consumer.queues[0]
        self.buffer = deque()
        self.consumer.register_callback(self._receive)

    def get(self, block=True, timeout=None):
        if not block:
            return self.get_nowait()
        self._consume()
        time_start = monotonic()
        remaining = timeout
        while True:
            if self.buffer:
                return self.buffer.popleft()
            if remaining is not None and remaining <= 0.0:
                raise self.Empty()
            try:
                self.channel.connection.client.drain_events(timeout=remaining)
            except socket.timeout:
                raise self.Empty()
            if remaining is not None:
                elapsed = monotonic() - time_start
                remaining = timeout - elapsed

    def get_nowait(self):
        m = self.queue.get(no_ack=self.no_ack, accept=self.consumer.accept)
        if not m:
            raise self.Empty()
        return m

    def put(self, message, serializer=None, headers=None, compression=None, routing_key=None, **kwargs):
        self.producer.publish(message, serializer=serializer, routing_key=routing_key, headers=headers, compression=compression, **kwargs)

    def clear(self):
        return self.consumer.purge()

    def qsize(self):
        _, size, _ = self.queue.queue_declare(passive=True)
        return size

    def close(self):
        self.consumer.cancel()

    def _receive(self, message_data, message):
        self.buffer.append(message)

    def _consume(self):
        if not self._consuming:
            self.consumer.consume(no_ack=self.no_ack)
            self._consuming = True

    def __len__(self):
        """`len(self) -> self.qsize()`."""
        return self.qsize()

    def __bool__(self):
        return True
    __nonzero__ = __bool__