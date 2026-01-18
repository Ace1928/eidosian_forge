from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import pickle
import sys
import threading
import time
from googlecloudsdk.core import exceptions
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import queue   # pylint: disable=redefined-builtin
from six.moves import range  # pylint: disable=redefined-builtin
class NullPool(BasePool):
    """Serial analog of parallel execution Pool."""

    def __init__(self):
        self._started = False

    def ApplyAsync(self, func, args):
        if not self._started:
            raise InvalidStateException('NullPool must be Start()ed before use.')
        try:
            result = _Result((func(*args),))
        except:
            result = _Result(exc_info=sys.exc_info())
        return _NullFuture(result)

    def Start(self):
        if self._started:
            raise InvalidStateException('Can only start NullPool once.')
        self._started = True

    def Join(self):
        if not self._started:
            raise InvalidStateException('NullPool must be Start()ed before use.')