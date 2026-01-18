from tornado.concurrent import Future
from tornado import gen
from tornado import netutil
from tornado.ioloop import IOLoop
from tornado.iostream import (
from tornado.httputil import HTTPHeaders
from tornado.locks import Condition, Event
from tornado.log import gen_log
from tornado.netutil import ssl_options_to_context, ssl_wrap_socket
from tornado.platform.asyncio import AddThreadSelectorEventLoop
from tornado.tcpserver import TCPServer
from tornado.testing import (
from tornado.test.util import (
from tornado.web import RequestHandler, Application
import asyncio
import errno
import hashlib
import logging
import os
import platform
import random
import socket
import ssl
import typing
from unittest import mock
import unittest
class TestReadWriteMixin(object):

    def make_iostream_pair(self, **kwargs):
        raise NotImplementedError

    def iostream_pair(self, **kwargs):
        """Like make_iostream_pair, but called by ``async with``.

        In py37 this becomes simpler with contextlib.asynccontextmanager.
        """

        class IOStreamPairContext:

            def __init__(self, test, kwargs):
                self.test = test
                self.kwargs = kwargs

            async def __aenter__(self):
                self.pair = await self.test.make_iostream_pair(**self.kwargs)
                return self.pair

            async def __aexit__(self, typ, value, tb):
                for s in self.pair:
                    s.close()
        return IOStreamPairContext(self, kwargs)

    @gen_test
    def test_write_zero_bytes(self):
        rs, ws = (yield self.make_iostream_pair())
        yield ws.write(b'')
        ws.close()
        rs.close()

    @gen_test
    def test_future_delayed_close_callback(self: typing.Any):
        rs, ws = (yield self.make_iostream_pair())
        try:
            ws.write(b'12')
            chunks = []
            chunks.append((yield rs.read_bytes(1)))
            ws.close()
            chunks.append((yield rs.read_bytes(1)))
            self.assertEqual(chunks, [b'1', b'2'])
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_close_buffered_data(self: typing.Any):
        rs, ws = (yield self.make_iostream_pair(read_chunk_size=256))
        try:
            ws.write(b'A' * 512)
            data = (yield rs.read_bytes(256))
            self.assertEqual(b'A' * 256, data)
            ws.close()
            yield gen.sleep(0.01)
            data = (yield rs.read_bytes(256))
            self.assertEqual(b'A' * 256, data)
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_read_until_close_after_close(self: typing.Any):
        rs, ws = (yield self.make_iostream_pair())
        try:
            ws.write(b'1234')
            data = (yield rs.read_bytes(1))
            ws.close()
            self.assertEqual(data, b'1')
            data = (yield rs.read_until_close())
            self.assertEqual(data, b'234')
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_large_read_until(self: typing.Any):
        rs, ws = (yield self.make_iostream_pair())
        try:
            if isinstance(rs, SSLIOStream) and platform.python_implementation() == 'PyPy':
                raise unittest.SkipTest('pypy gc causes problems with openssl')
            NUM_KB = 4096
            for i in range(NUM_KB):
                ws.write(b'A' * 1024)
            ws.write(b'\r\n')
            data = (yield rs.read_until(b'\r\n'))
            self.assertEqual(len(data), NUM_KB * 1024 + 2)
        finally:
            ws.close()
            rs.close()

    @gen_test
    async def test_read_until_with_close_after_second_packet(self):
        async with self.iostream_pair() as (rs, ws):
            rf = asyncio.ensure_future(rs.read_until(b'done'))
            await asyncio.sleep(0.1)
            await ws.write(b'x' * 2048)
            ws.write(b'done')
            ws.close()
            await rf

    @gen_test
    async def test_read_until_unsatisfied_after_close(self: typing.Any):
        async with self.iostream_pair() as (rs, ws):
            rf = asyncio.ensure_future(rs.read_until(b'done'))
            await ws.write(b'x' * 2048)
            ws.write(b'foo')
            ws.close()
            with self.assertRaises(StreamClosedError):
                await rf

    @gen_test
    def test_close_callback_with_pending_read(self: typing.Any):
        OK = b'OK\r\n'
        rs, ws = (yield self.make_iostream_pair())
        event = Event()
        rs.set_close_callback(event.set)
        try:
            ws.write(OK)
            res = (yield rs.read_until(b'\r\n'))
            self.assertEqual(res, OK)
            ws.close()
            rs.read_until(b'\r\n')
            yield event.wait()
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_future_close_callback(self: typing.Any):
        rs, ws = (yield self.make_iostream_pair())
        closed = [False]
        cond = Condition()

        def close_callback():
            closed[0] = True
            cond.notify()
        rs.set_close_callback(close_callback)
        try:
            ws.write(b'a')
            res = (yield rs.read_bytes(1))
            self.assertEqual(res, b'a')
            self.assertFalse(closed[0])
            ws.close()
            yield cond.wait()
            self.assertTrue(closed[0])
        finally:
            rs.close()
            ws.close()

    @gen_test
    def test_write_memoryview(self: typing.Any):
        rs, ws = (yield self.make_iostream_pair())
        try:
            fut = rs.read_bytes(4)
            ws.write(memoryview(b'hello'))
            data = (yield fut)
            self.assertEqual(data, b'hell')
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_read_bytes_partial(self: typing.Any):
        rs, ws = (yield self.make_iostream_pair())
        try:
            fut = rs.read_bytes(50, partial=True)
            ws.write(b'hello')
            data = (yield fut)
            self.assertEqual(data, b'hello')
            fut = rs.read_bytes(3, partial=True)
            ws.write(b'world')
            data = (yield fut)
            self.assertEqual(data, b'wor')
            data = (yield rs.read_bytes(0, partial=True))
            self.assertEqual(data, b'')
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_read_until_max_bytes(self: typing.Any):
        rs, ws = (yield self.make_iostream_pair())
        closed = Event()
        rs.set_close_callback(closed.set)
        try:
            fut = rs.read_until(b'def', max_bytes=50)
            ws.write(b'abcdef')
            data = (yield fut)
            self.assertEqual(data, b'abcdef')
            fut = rs.read_until(b'def', max_bytes=6)
            ws.write(b'abcdef')
            data = (yield fut)
            self.assertEqual(data, b'abcdef')
            with ExpectLog(gen_log, 'Unsatisfiable read', level=logging.INFO):
                fut = rs.read_until(b'def', max_bytes=5)
                ws.write(b'123456')
                yield closed.wait()
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_read_until_max_bytes_inline(self: typing.Any):
        rs, ws = (yield self.make_iostream_pair())
        closed = Event()
        rs.set_close_callback(closed.set)
        try:
            ws.write(b'123456')
            with ExpectLog(gen_log, 'Unsatisfiable read', level=logging.INFO):
                with self.assertRaises(StreamClosedError):
                    yield rs.read_until(b'def', max_bytes=5)
            yield closed.wait()
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_read_until_max_bytes_ignores_extra(self: typing.Any):
        rs, ws = (yield self.make_iostream_pair())
        closed = Event()
        rs.set_close_callback(closed.set)
        try:
            ws.write(b'abcdef')
            with ExpectLog(gen_log, 'Unsatisfiable read', level=logging.INFO):
                rs.read_until(b'def', max_bytes=5)
                yield closed.wait()
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_read_until_regex_max_bytes(self: typing.Any):
        rs, ws = (yield self.make_iostream_pair())
        closed = Event()
        rs.set_close_callback(closed.set)
        try:
            fut = rs.read_until_regex(b'def', max_bytes=50)
            ws.write(b'abcdef')
            data = (yield fut)
            self.assertEqual(data, b'abcdef')
            fut = rs.read_until_regex(b'def', max_bytes=6)
            ws.write(b'abcdef')
            data = (yield fut)
            self.assertEqual(data, b'abcdef')
            with ExpectLog(gen_log, 'Unsatisfiable read', level=logging.INFO):
                rs.read_until_regex(b'def', max_bytes=5)
                ws.write(b'123456')
                yield closed.wait()
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_read_until_regex_max_bytes_inline(self: typing.Any):
        rs, ws = (yield self.make_iostream_pair())
        closed = Event()
        rs.set_close_callback(closed.set)
        try:
            ws.write(b'123456')
            with ExpectLog(gen_log, 'Unsatisfiable read', level=logging.INFO):
                rs.read_until_regex(b'def', max_bytes=5)
                yield closed.wait()
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_read_until_regex_max_bytes_ignores_extra(self):
        rs, ws = (yield self.make_iostream_pair())
        closed = Event()
        rs.set_close_callback(closed.set)
        try:
            ws.write(b'abcdef')
            with ExpectLog(gen_log, 'Unsatisfiable read', level=logging.INFO):
                rs.read_until_regex(b'def', max_bytes=5)
                yield closed.wait()
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_small_reads_from_large_buffer(self: typing.Any):
        rs, ws = (yield self.make_iostream_pair(max_buffer_size=10 * 1024))
        try:
            ws.write(b'a' * 1024 * 100)
            for i in range(100):
                data = (yield rs.read_bytes(1024))
                self.assertEqual(data, b'a' * 1024)
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_small_read_untils_from_large_buffer(self: typing.Any):
        rs, ws = (yield self.make_iostream_pair(max_buffer_size=10 * 1024))
        try:
            ws.write((b'a' * 1023 + b'\n') * 100)
            for i in range(100):
                data = (yield rs.read_until(b'\n', max_bytes=4096))
                self.assertEqual(data, b'a' * 1023 + b'\n')
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_flow_control(self):
        MB = 1024 * 1024
        rs, ws = (yield self.make_iostream_pair(max_buffer_size=5 * MB))
        try:
            ws.write(b'a' * 10 * MB)
            yield rs.read_bytes(MB)
            yield gen.sleep(0.1)
            for i in range(9):
                yield rs.read_bytes(MB)
        finally:
            rs.close()
            ws.close()

    @gen_test
    def test_read_into(self: typing.Any):
        rs, ws = (yield self.make_iostream_pair())

        def sleep_some():
            self.io_loop.run_sync(lambda: gen.sleep(0.05))
        try:
            buf = bytearray(10)
            fut = rs.read_into(buf)
            ws.write(b'hello')
            yield gen.sleep(0.05)
            self.assertTrue(rs.reading())
            ws.write(b'world!!')
            data = (yield fut)
            self.assertFalse(rs.reading())
            self.assertEqual(data, 10)
            self.assertEqual(bytes(buf), b'helloworld')
            fut = rs.read_into(buf)
            yield gen.sleep(0.05)
            self.assertTrue(rs.reading())
            ws.write(b'1234567890')
            data = (yield fut)
            self.assertFalse(rs.reading())
            self.assertEqual(data, 10)
            self.assertEqual(bytes(buf), b'!!12345678')
            buf = bytearray(4)
            ws.write(b'abcdefghi')
            data = (yield rs.read_into(buf))
            self.assertEqual(data, 4)
            self.assertEqual(bytes(buf), b'90ab')
            data = (yield rs.read_bytes(7))
            self.assertEqual(data, b'cdefghi')
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_read_into_partial(self: typing.Any):
        rs, ws = (yield self.make_iostream_pair())
        try:
            buf = bytearray(10)
            fut = rs.read_into(buf, partial=True)
            ws.write(b'hello')
            data = (yield fut)
            self.assertFalse(rs.reading())
            self.assertEqual(data, 5)
            self.assertEqual(bytes(buf), b'hello\x00\x00\x00\x00\x00')
            ws.write(b'world!1234567890')
            data = (yield rs.read_into(buf, partial=True))
            self.assertEqual(data, 10)
            self.assertEqual(bytes(buf), b'world!1234')
            data = (yield rs.read_into(buf, partial=True))
            self.assertEqual(data, 6)
            self.assertEqual(bytes(buf), b'5678901234')
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_read_into_zero_bytes(self: typing.Any):
        rs, ws = (yield self.make_iostream_pair())
        try:
            buf = bytearray()
            fut = rs.read_into(buf)
            self.assertEqual(fut.result(), 0)
        finally:
            ws.close()
            rs.close()

    @gen_test
    def test_many_mixed_reads(self):
        r = random.Random(42)
        nbytes = 1000000
        rs, ws = (yield self.make_iostream_pair())
        produce_hash = hashlib.sha1()
        consume_hash = hashlib.sha1()

        @gen.coroutine
        def produce():
            remaining = nbytes
            while remaining > 0:
                size = r.randint(1, min(1000, remaining))
                data = os.urandom(size)
                produce_hash.update(data)
                yield ws.write(data)
                remaining -= size
            assert remaining == 0

        @gen.coroutine
        def consume():
            remaining = nbytes
            while remaining > 0:
                if r.random() > 0.5:
                    size = r.randint(1, min(1000, remaining))
                    data = (yield rs.read_bytes(size))
                    consume_hash.update(data)
                    remaining -= size
                else:
                    size = r.randint(1, min(1000, remaining))
                    buf = bytearray(size)
                    n = (yield rs.read_into(buf))
                    assert n == size
                    consume_hash.update(buf)
                    remaining -= size
            assert remaining == 0
        try:
            yield [produce(), consume()]
            assert produce_hash.hexdigest() == consume_hash.hexdigest()
        finally:
            ws.close()
            rs.close()