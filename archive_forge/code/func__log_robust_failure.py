import asyncio
import logging
import threading
import weakref
from asgiref.sync import async_to_sync, iscoroutinefunction, sync_to_async
from django.utils.inspect import func_accepts_kwargs
def _log_robust_failure(self, receiver, err):
    logger.error('Error calling %s in Signal.send_robust() (%s)', receiver.__qualname__, err, exc_info=err)