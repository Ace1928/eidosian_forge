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
class BaseFuture(object):
    """A future object containing a value that may not be available yet."""

    def Get(self):
        return self.GetResult().GetOrRaise()

    @abc.abstractmethod
    def GetResult(self):
        raise NotImplementedError

    @abc.abstractmethod
    def Done(self):
        raise NotImplementedError