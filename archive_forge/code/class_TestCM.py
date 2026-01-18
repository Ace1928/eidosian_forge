import asyncio
import contextvars
import unittest
from test import support
class TestCM:

    def __init__(self, ordering, enter_result=None):
        self.ordering = ordering
        self.enter_result = enter_result

    async def __aenter__(self):
        self.ordering.append('enter')
        return self.enter_result

    async def __aexit__(self, *exc_info):
        self.ordering.append('exit')