import collections
import fractions
import functools
import heapq
import inspect
import logging
import math
import random
import threading
from concurrent import futures
import futurist
from futurist import _utils as utils
class ExecutorFactory(object):
    """Base class for any executor factory."""
    shutdown = True
    'Whether the executor should be shut down on periodic worker stop.'

    def __call__(self):
        """Return the executor to be used."""
        raise NotImplementedError()