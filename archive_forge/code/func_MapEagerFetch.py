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