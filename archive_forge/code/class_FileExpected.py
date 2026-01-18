from __future__ import print_function, unicode_literals
import typing
import functools
import six
from six import text_type
class FileExpected(ResourceInvalid):
    """Operation only works on files."""
    default_message = "path '{path}' should be a file"