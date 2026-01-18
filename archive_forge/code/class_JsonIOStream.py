from __future__ import annotations
import collections
import contextlib
import functools
import itertools
import os
import socket
import sys
import threading
from debugpy.common import json, log, util
from debugpy.common.util import hide_thread_from_debugger
class JsonIOStream(object):
    """Implements a JSON value stream over two byte streams (input and output).

    Each value is encoded as a DAP packet, with metadata headers and a JSON payload.
    """
    MAX_BODY_SIZE = 16777215
    json_decoder_factory = json.JsonDecoder
    'Used by read_json() when decoder is None.'
    json_encoder_factory = json.JsonEncoder
    'Used by write_json() when encoder is None.'

    @classmethod
    def from_stdio(cls, name='stdio'):
        """Creates a new instance that receives messages from sys.stdin, and sends
        them to sys.stdout.
        """
        return cls(sys.stdin.buffer, sys.stdout.buffer, name)

    @classmethod
    def from_process(cls, process, name='stdio'):
        """Creates a new instance that receives messages from process.stdin, and sends
        them to process.stdout.
        """
        return cls(process.stdout, process.stdin, name)

    @classmethod
    def from_socket(cls, sock, name=None):
        """Creates a new instance that sends and receives messages over a socket."""
        sock.settimeout(None)
        if name is None:
            name = repr(sock)
        socket_io = sock.makefile('rwb', 0)

        def cleanup():
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            sock.close()
        return cls(socket_io, socket_io, name, cleanup)

    def __init__(self, reader, writer, name=None, cleanup=lambda: None):
        """Creates a new JsonIOStream.

        reader must be a BytesIO-like object, from which incoming messages will be
        read by read_json().

        writer must be a BytesIO-like object, into which outgoing messages will be
        written by write_json().

        cleanup must be a callable; it will be invoked without arguments when the
        stream is closed.

        reader.readline() must treat "
" as the line terminator, and must leave "\r"
        as is - it must not replace "\r
" with "
" automatically, as TextIO does.
        """
        if name is None:
            name = f'reader={reader!r}, writer={writer!r}'
        self.name = name
        self._reader = reader
        self._writer = writer
        self._cleanup = cleanup
        self._closed = False

    def close(self):
        """Closes the stream, the reader, and the writer."""
        if self._closed:
            return
        self._closed = True
        log.debug('Closing {0} message stream', self.name)
        try:
            try:
                try:
                    self._writer.close()
                finally:
                    if self._reader is not self._writer:
                        self._reader.close()
            finally:
                self._cleanup()
        except Exception:
            log.reraise_exception('Error while closing {0} message stream', self.name)

    def _log_message(self, dir, data, logger=log.debug):
        return logger('{0} {1} {2}', self.name, dir, data)

    def _read_line(self, reader):
        line = b''
        while True:
            try:
                line += reader.readline()
            except Exception as exc:
                raise NoMoreMessages(str(exc), stream=self)
            if not line:
                raise NoMoreMessages(stream=self)
            if line.endswith(b'\r\n'):
                line = line[0:-2]
                return line

    def read_json(self, decoder=None):
        """Read a single JSON value from reader.

        Returns JSON value as parsed by decoder.decode(), or raises NoMoreMessages
        if there are no more values to be read.
        """
        decoder = decoder if decoder is not None else self.json_decoder_factory()
        reader = self._reader
        read_line = functools.partial(self._read_line, reader)

        def log_message_and_reraise_exception(format_string='', *args, **kwargs):
            if format_string:
                format_string += '\n\n'
            format_string += '{name} -->\n{raw_lines}'
            raw_lines = b''.join(raw_chunks).split(b'\n')
            raw_lines = '\n'.join((repr(line) for line in raw_lines))
            log.reraise_exception(format_string, *args, name=self.name, raw_lines=raw_lines, **kwargs)
        raw_chunks = []
        headers = {}
        while True:
            try:
                line = read_line()
            except Exception:
                if headers:
                    log_message_and_reraise_exception('Error while reading message headers:')
                else:
                    raise
            raw_chunks += [line, b'\n']
            if line == b'':
                break
            key, _, value = line.partition(b':')
            headers[key] = value
        try:
            length = int(headers[b'Content-Length'])
            if not 0 <= length <= self.MAX_BODY_SIZE:
                raise ValueError
        except (KeyError, ValueError):
            try:
                raise IOError('Content-Length is missing or invalid:')
            except Exception:
                log_message_and_reraise_exception()
        body_start = len(raw_chunks)
        body_remaining = length
        while body_remaining > 0:
            try:
                chunk = reader.read(body_remaining)
                if not chunk:
                    raise EOFError
            except Exception as exc:
                raise NoMoreMessages(str(exc), stream=self)
            raw_chunks.append(chunk)
            body_remaining -= len(chunk)
        assert body_remaining == 0
        body = b''.join(raw_chunks[body_start:])
        try:
            body = body.decode('utf-8')
        except Exception:
            log_message_and_reraise_exception()
        try:
            body = decoder.decode(body)
        except Exception:
            log_message_and_reraise_exception()
        self._log_message('-->', body)
        return body

    def write_json(self, value, encoder=None):
        """Write a single JSON value into writer.

        Value is written as encoded by encoder.encode().
        """
        if self._closed:
            raise NoMoreMessages(stream=self)
        encoder = encoder if encoder is not None else self.json_encoder_factory()
        writer = self._writer
        try:
            body = encoder.encode(value)
        except Exception:
            self._log_message('<--', repr(value), logger=log.reraise_exception)
        body = body.encode('utf-8')
        header = f'Content-Length: {len(body)}\r\n\r\n'.encode('ascii')
        data = header + body
        data_written = 0
        try:
            while data_written < len(data):
                written = writer.write(data[data_written:])
                data_written += written
            writer.flush()
        except Exception as exc:
            self._log_message('<--', value, logger=log.swallow_exception)
            raise JsonIOError(stream=self, cause=exc)
        self._log_message('<--', value)

    def __repr__(self):
        return f'{type(self).__name__}({self.name!r})'