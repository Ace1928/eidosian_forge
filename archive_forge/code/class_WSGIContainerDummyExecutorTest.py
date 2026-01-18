import asyncio
import concurrent.futures
import threading
from wsgiref.validate import validator
from tornado.routing import RuleRouter
from tornado.testing import AsyncHTTPTestCase, gen_test
from tornado.wsgi import WSGIContainer
class WSGIContainerDummyExecutorTest(WSGIAppMixin, AsyncHTTPTestCase):

    def get_executor(self):
        return None

    def test_simple(self):
        response = self.fetch('/simple')
        self.assertEqual(response.body, b'Hello world!')

    @gen_test
    async def test_concurrent_barrier(self):
        self.barrier.reset()
        resps = await asyncio.gather(self.http_client.fetch(self.get_url('/barrier')), self.http_client.fetch(self.get_url('/barrier')))
        for resp in resps:
            self.assertEqual(resp.body, b'broken barrier')

    @gen_test
    async def test_concurrent_streaming_barrier(self):
        self.barrier.reset()
        resps = await asyncio.gather(self.http_client.fetch(self.get_url('/streaming_barrier')), self.http_client.fetch(self.get_url('/streaming_barrier')))
        for resp in resps:
            self.assertEqual(resp.body, b'ok broken barrier')