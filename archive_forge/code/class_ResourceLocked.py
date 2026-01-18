from __future__ import print_function, unicode_literals
import typing
import functools
import six
from six import text_type
class ResourceLocked(ResourceError):
    """Attempt to use a locked resource."""
    default_message = "resource '{path}' is locked"