import collections.abc
import datetime
import email.utils
import functools
import logging
import io
import re
import subprocess
import warnings
import chardet
from debian._util import (
from debian.deprecation import function_deprecated_by
import debian.debian_support
import debian.changelog
class RestrictedField(collections.namedtuple('RestrictedField', 'name from_str to_str allow_none')):
    """Placeholder for a property providing access to a restricted field.

    Use this as an attribute when defining a subclass of RestrictedWrapper.
    It will be replaced with a property.  See the RestrictedWrapper
    documentation for an example.
    """

    def __new__(cls, name, from_str=None, to_str=None, allow_none=True):
        """Create a new RestrictedField placeholder.

        The getter that will replace this returns (or applies the given to_str
        function to) None for fields that do not exist in the underlying data
        object.

        :param name: The name of the deb822 field.
        :param from_str: The function to apply for getters (default is to
            return the string directly).
        :param to_str: The function to apply for setters (default is to use the
            value directly).  If allow_none is True, this function may return
            None, in which case the underlying key is deleted.
        :param allow_none: Whether it is allowed to set the value to None
            (which results in the underlying key being deleted).
        """
        return super(RestrictedField, cls).__new__(cls, name, from_str=from_str, to_str=to_str, allow_none=allow_none)