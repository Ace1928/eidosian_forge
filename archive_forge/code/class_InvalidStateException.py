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
class InvalidStateException(Exception):
    """Exception to indicate that a parallel pool was put in an invalid state."""

    def __init__(self, msg):
        super(InvalidStateException, self).__init__(msg)