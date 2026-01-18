import asyncio
import asyncio.events as events
import os
import sys
import threading
from contextlib import contextmanager, suppress
from heapq import heappop
def _patch_policy():
    """Patch the policy to always return a patched loop."""

    def get_event_loop(self):
        if self._local._loop is None:
            loop = self.new_event_loop()
            _patch_loop(loop)
            self.set_event_loop(loop)
        return self._local._loop
    policy = events.get_event_loop_policy()
    policy.__class__.get_event_loop = get_event_loop