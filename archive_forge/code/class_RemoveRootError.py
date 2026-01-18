from __future__ import print_function, unicode_literals
import typing
import functools
import six
from six import text_type
class RemoveRootError(OperationFailed):
    """Attempt to remove the root directory."""
    default_message = 'root directory may not be removed'