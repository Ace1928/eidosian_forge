import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
def _check_key(key):
    """Helper function to assert that a key is properly formatted.

    This generally shouldn't be used in production code, but it can be helpful
    to debug problems.
    """
    if not isinstance(key, StaticTuple):
        raise TypeError('key {!r} is not StaticTuple but {}'.format(key, type(key)))
    if len(key) != 1:
        raise ValueError('key %r should have length 1, not %d' % (key, len(key)))
    if not isinstance(key[0], str):
        raise TypeError('key %r should hold a str, not %r' % (key, type(key[0])))
    if not key[0].startswith('sha1:'):
        raise ValueError('key {!r} should point to a sha1:'.format(key))