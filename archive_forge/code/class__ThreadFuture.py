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
class _ThreadFuture(BaseFuture):

    def __init__(self, task, results_map):
        self._task = task
        self._results_map = results_map

    def Get(self):
        """Return the value of the future, or raise an exception."""
        return self.GetResult().GetOrRaise()

    def GetResult(self):
        """Get the _Result of the future."""
        while True:
            if self._task in self._results_map:
                return self._results_map[self._task]
            time.sleep(_POLL_INTERVAL)

    def Done(self):
        """Return True if the task finished with or without errors."""
        return self._task in self._results_map