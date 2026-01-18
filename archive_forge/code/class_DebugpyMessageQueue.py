import os
import re
import sys
import typing as t
from pathlib import Path
import zmq
from IPython.core.getipython import get_ipython
from IPython.core.inputtransformer2 import leading_empty_lines
from tornado.locks import Event
from tornado.queues import Queue
from zmq.utils import jsonapi
from .compiler import get_file_name, get_tmp_directory, get_tmp_hash_seed
class DebugpyMessageQueue:
    """A debugpy message queue."""
    HEADER = 'Content-Length: '
    HEADER_LENGTH = 16
    SEPARATOR = '\r\n\r\n'
    SEPARATOR_LENGTH = 4

    def __init__(self, event_callback, log):
        """Init the queue."""
        self.tcp_buffer = ''
        self._reset_tcp_pos()
        self.event_callback = event_callback
        self.message_queue: Queue[t.Any] = Queue()
        self.log = log

    def _reset_tcp_pos(self):
        self.header_pos = -1
        self.separator_pos = -1
        self.message_size = 0
        self.message_pos = -1

    def _put_message(self, raw_msg):
        self.log.debug('QUEUE - _put_message:')
        msg = t.cast(t.Dict[str, t.Any], jsonapi.loads(raw_msg))
        if msg['type'] == 'event':
            self.log.debug('QUEUE - received event:')
            self.log.debug(msg)
            self.event_callback(msg)
        else:
            self.log.debug('QUEUE - put message:')
            self.log.debug(msg)
            self.message_queue.put_nowait(msg)

    def put_tcp_frame(self, frame):
        """Put a tcp frame in the queue."""
        self.tcp_buffer += frame
        self.log.debug('QUEUE - received frame')
        while True:
            if self.header_pos == -1:
                self.header_pos = self.tcp_buffer.find(DebugpyMessageQueue.HEADER)
            if self.header_pos == -1:
                return
            self.log.debug('QUEUE - found header at pos %i', self.header_pos)
            if self.separator_pos == -1:
                hint = self.header_pos + DebugpyMessageQueue.HEADER_LENGTH
                self.separator_pos = self.tcp_buffer.find(DebugpyMessageQueue.SEPARATOR, hint)
            if self.separator_pos == -1:
                return
            self.log.debug('QUEUE - found separator at pos %i', self.separator_pos)
            if self.message_pos == -1:
                size_pos = self.header_pos + DebugpyMessageQueue.HEADER_LENGTH
                self.message_pos = self.separator_pos + DebugpyMessageQueue.SEPARATOR_LENGTH
                self.message_size = int(self.tcp_buffer[size_pos:self.separator_pos])
            self.log.debug('QUEUE - found message at pos %i', self.message_pos)
            self.log.debug('QUEUE - message size is %i', self.message_size)
            if len(self.tcp_buffer) - self.message_pos < self.message_size:
                return
            self._put_message(self.tcp_buffer[self.message_pos:self.message_pos + self.message_size])
            if len(self.tcp_buffer) - self.message_pos == self.message_size:
                self.log.debug('QUEUE - resetting tcp_buffer')
                self.tcp_buffer = ''
                self._reset_tcp_pos()
                return
            self.tcp_buffer = self.tcp_buffer[self.message_pos + self.message_size:]
            self.log.debug('QUEUE - slicing tcp_buffer: %s', self.tcp_buffer)
            self._reset_tcp_pos()

    async def get_message(self):
        """Get a message from the queue."""
        return await self.message_queue.get()