from __future__ import print_function, unicode_literals
import typing
import functools
import six
from six import text_type
class InsufficientStorage(OperationFailed):
    """Storage is insufficient for requested operation."""
    default_message = 'insufficient storage space'