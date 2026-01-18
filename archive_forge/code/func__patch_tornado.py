import asyncio
import asyncio.events as events
import os
import sys
import threading
from contextlib import contextmanager, suppress
from heapq import heappop
def _patch_tornado():
    """
    If tornado is imported before nest_asyncio, make tornado aware of
    the pure-Python asyncio Future.
    """
    if 'tornado' in sys.modules:
        import tornado.concurrent as tc
        tc.Future = asyncio.Future
        if asyncio.Future not in tc.FUTURES:
            tc.FUTURES += (asyncio.Future,)