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
@six.add_metaclass(abc.ABCMeta)
class BasePool(object):
    """Base class for parallel pools.

  Provides a limited subset of the multiprocessing.Pool API.

  Can be used as a context manager:

  >>> with pool:
  ...  assert pool.Map(str, [1, 2, 3]) == ['1', '2', '3']
  """

    @abc.abstractmethod
    def Start(self):
        """Initialize non-trivial infrastructure (e.g. processes/threads/queues)."""
        raise NotImplementedError

    @abc.abstractmethod
    def Join(self):
        """Clean up anything started in Start()."""
        raise NotImplementedError

    def Map(self, func, iterable):
        """Applies func to each element in iterable and return a list of results."""
        return self.MapAsync(func, iterable).Get()

    def MapAsync(self, func, iterable):
        """Applies func to each element in iterable and return a future."""
        return _MultiFuture([self.ApplyAsync(func, (arg,)) for arg in iterable])

    def MapEagerFetch(self, func, iterable):
        """Applies func to each element in iterable and return a generator.

    The generator yields the result immediately after the task is done. So
    result for faster task will be yielded earlier than for slower task.

    Args:
      func: a function object
      iterable: an iterable object and each element is the arguments to func

    Returns:
      A generator to produce the results.
    """
        return self.MapAsync(func, iterable).GetResultsEagerFetch()

    def Apply(self, func, args):
        """Applies func to args and returns the result."""
        return self.ApplyAsync(func, args).Get()

    @abc.abstractmethod
    def ApplyAsync(self, func, args):
        """Apply func to args and return a future."""
        raise NotImplementedError

    def __enter__(self):
        self.Start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.Join()