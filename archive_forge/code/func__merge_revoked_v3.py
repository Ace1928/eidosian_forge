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
def _merge_revoked_v3(self, zrevoked):
    if zrevoked:
        self._revoked_tasks.update(pickle.loads(self.decompress(zrevoked)))