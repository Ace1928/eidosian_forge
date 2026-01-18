import asyncio
import contextvars
import unittest
from test import support
class TestCase1(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        asyncio.get_event_loop_policy().get_event_loop()

    async def test_demo1(self):
        pass