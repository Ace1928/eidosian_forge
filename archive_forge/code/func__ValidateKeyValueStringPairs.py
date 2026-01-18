from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import posixpath
import random
import re
import string
import sys
from googlecloudsdk.api_lib.firebase.test import exceptions as test_exceptions
from googlecloudsdk.api_lib.firebase.test import util as util
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import files
import six
def _ValidateKeyValueStringPairs(arg_internal_name, arg_value):
    """Validates that an argument is a dict of string-type key-value pairs."""
    if isinstance(arg_value, dict):
        new_dict = {}
        for key, value in arg_value.items():
            new_dict[str(key)] = _ValidateString(arg_internal_name, value)
        return new_dict
    else:
        raise test_exceptions.InvalidArgException(arg_internal_name, 'Malformed key-value pairs.')