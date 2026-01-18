from collections import defaultdict
from functools import partial
from heapq import heappush
from operator import itemgetter
from kombu import Consumer
from kombu.asynchronous.semaphore import DummyLock
from kombu.exceptions import ContentDisallowed, DecodeError
from celery import bootsteps
from celery.utils.log import get_logger
from celery.utils.objects import Bunch
from .mingle import Mingle
def _call_handlers(self, handlers, *args, **kwargs):
    for handler in handlers:
        try:
            handler(*args, **kwargs)
        except Exception as exc:
            logger.exception('Ignored error from handler %r: %r', handler, exc)