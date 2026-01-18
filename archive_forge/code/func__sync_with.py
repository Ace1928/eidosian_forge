import os
import platform
import shelve
import sys
import weakref
import zlib
from collections import Counter
from kombu.serialization import pickle, pickle_protocol
from kombu.utils.objects import cached_property
from celery import __version__
from celery.exceptions import WorkerShutdown, WorkerTerminate
from celery.utils.collections import LimitedSet
def _sync_with(self, d):
    self._revoked_tasks.purge()
    d.update({'__proto__': 3, 'zrevoked': self.compress(self._dumps(self._revoked_tasks)), 'clock': self.clock.forward() if self.clock else 0})
    return d