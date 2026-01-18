import inspect
from collections import UserList
from functools import partial
from itertools import islice, tee, zip_longest
from typing import Any, Callable
from kombu.utils.functional import LRUCache, dictfilter, is_list, lazy, maybe_evaluate, maybe_list, memoize
from vine import promise
from celery.utils.log import get_logger
class mlazy(lazy):
    """Memoized lazy evaluation.

    The function is only evaluated once, every subsequent access
    will return the same value.
    """
    evaluated = False
    _value = None

    def evaluate(self):
        if not self.evaluated:
            self._value = super().evaluate()
            self.evaluated = True
        return self._value