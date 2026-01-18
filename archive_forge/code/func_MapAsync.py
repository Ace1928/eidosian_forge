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
def MapAsync(self, func, iterable):
    """Applies func to each element in iterable and return a future."""
    return _MultiFuture([self.ApplyAsync(func, (arg,)) for arg in iterable])