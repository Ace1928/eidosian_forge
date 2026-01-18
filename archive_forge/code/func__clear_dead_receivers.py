import asyncio
import logging
import threading
import weakref
from asgiref.sync import async_to_sync, iscoroutinefunction, sync_to_async
from django.utils.inspect import func_accepts_kwargs
def _clear_dead_receivers(self):
    if self._dead_receivers:
        self._dead_receivers = False
        self.receivers = [r for r in self.receivers if not (isinstance(r[1], weakref.ReferenceType) and r[1]() is None)]