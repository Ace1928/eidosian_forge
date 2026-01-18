from __future__ import print_function, unicode_literals
import typing
import functools
import six
from six import text_type
class DestinationExists(ResourceError):
    """Target destination already exists."""
    default_message = "destination '{path}' exists"