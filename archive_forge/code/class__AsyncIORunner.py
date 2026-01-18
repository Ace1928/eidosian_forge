import ast
import asyncio
import inspect
from functools import wraps
class _AsyncIORunner:

    def __call__(self, coro):
        """
        Handler for asyncio autoawait
        """
        return get_asyncio_loop().run_until_complete(coro)

    def __str__(self):
        return 'asyncio'