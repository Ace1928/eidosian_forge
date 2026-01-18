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
def GetOrRaise(self):
    if self.value:
        return self.value[0]
    elif self.error:
        raise self.error
    else:
        exceptions.reraise(self.exc_info[1], tb=self.exc_info[2])