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
class _Task(object):
    """An individual work unit to be performed in parallel.

  Attributes:
    func: callable, a function to be called with the given arguments. Must be
      serializable.
    args: tuple, the arguments to pass to func. Must be serializable.
  """

    def __init__(self, func, args):
        self.func = func
        self.args = args

    def __hash__(self):
        return hash((self.func.__name__, self.args))

    def __eq__(self, other):
        return self.func.__name__ == other.func.__name__ and self.args == other.args

    def __ne__(self, other):
        return not self.__eq__(other)