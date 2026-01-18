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
class MultiError(Exception):

    def __init__(self, errors):
        self.errors = errors
        fn = lambda e: '{}: {}'.format(type(e).__name__, six.text_type(e))
        super(MultiError, self).__init__('One or more errors occurred:\n' + '\n\n'.join(map(fn, errors)))