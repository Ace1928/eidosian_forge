from __future__ import print_function, unicode_literals
import typing
import functools
import six
from six import text_type
class OperationTimeout(OperationFailed):
    """Filesystem took too long."""
    default_message = 'operation timed out'