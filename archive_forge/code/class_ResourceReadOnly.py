from __future__ import print_function, unicode_literals
import typing
import functools
import six
from six import text_type
class ResourceReadOnly(ResourceError):
    """Attempting to modify a read-only resource."""
    default_message = "resource '{path}' is read only"