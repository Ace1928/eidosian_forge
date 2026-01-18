import asyncio
import contextlib
import gc
import io
import sys
import traceback
import types
import typing
import unittest
import tornado
from tornado import web, gen, httpclient
from tornado.test.util import skipNotCPython
@skipNotCPython
class CircleRefsTest(unittest.TestCase):

    def test_known_leak(self):

        class C(object):

            def __init__(self, name):
                self.name = name
                self.a: typing.Optional[C] = None
                self.b: typing.Optional[C] = None
                self.c: typing.Optional[C] = None

            def __repr__(self):
                return f'name={self.name}'
        with self.assertRaises(AssertionError) as cm:
            with assert_no_cycle_garbage():
                a = C('a')
                b = C('b')
                c = C('c')
                a.b = b
                a.c = c
                b.a = a
                b.c = c
                del a, b
        self.assertIn('Circular', str(cm.exception))
        self.assertIn('    name=a', str(cm.exception))
        self.assertIn('    name=b', str(cm.exception))
        self.assertNotIn('    name=c', str(cm.exception))

    async def run_handler(self, handler_class):
        app = web.Application([('/', handler_class)])
        socket, port = tornado.testing.bind_unused_port()
        server = tornado.httpserver.HTTPServer(app)
        server.add_socket(socket)
        client = httpclient.AsyncHTTPClient()
        with assert_no_cycle_garbage():
            await client.fetch(f'http://127.0.0.1:{port}/')
        client.close()
        server.stop()
        socket.close()

    def test_sync_handler(self):

        class Handler(web.RequestHandler):

            def get(self):
                self.write('ok\n')
        asyncio.run(self.run_handler(Handler))

    def test_finish_exception_handler(self):

        class Handler(web.RequestHandler):

            def get(self):
                raise web.Finish('ok\n')
        asyncio.run(self.run_handler(Handler))

    def test_coro_handler(self):

        class Handler(web.RequestHandler):

            @gen.coroutine
            def get(self):
                yield asyncio.sleep(0.01)
                self.write('ok\n')
        asyncio.run(self.run_handler(Handler))

    def test_async_handler(self):

        class Handler(web.RequestHandler):

            async def get(self):
                await asyncio.sleep(0.01)
                self.write('ok\n')
        asyncio.run(self.run_handler(Handler))

    def test_run_on_executor(self):
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(1) as thread_pool:

            class Factory(object):
                executor = thread_pool

                @tornado.concurrent.run_on_executor
                def run(self):
                    return None
            factory = Factory()

            async def main():
                for i in range(2):
                    await factory.run()
            with assert_no_cycle_garbage():
                asyncio.run(main())