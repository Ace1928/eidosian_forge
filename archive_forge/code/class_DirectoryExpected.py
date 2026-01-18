from __future__ import print_function, unicode_literals
import typing
import functools
import six
from six import text_type
class DirectoryExpected(ResourceInvalid):
    """Operation only works on directories."""
    default_message = "path '{path}' should be a directory"