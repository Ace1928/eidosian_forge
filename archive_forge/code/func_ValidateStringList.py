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
def ValidateStringList(arg_internal_name, arg_value):
    """Validates an arg whose value should be a list of strings.

  Args:
    arg_internal_name: the internal form of the arg name.
    arg_value: the argument's value parsed from yaml file.

  Returns:
    The validated argument value.

  Raises:
    InvalidArgException: the argument's value is not valid.
  """
    if isinstance(arg_value, six.string_types):
        return [arg_value]
    if isinstance(arg_value, int):
        return [str(arg_value)]
    if isinstance(arg_value, list):
        return [_ValidateString(arg_internal_name, value) for value in arg_value]
    raise test_exceptions.InvalidArgException(arg_internal_name, arg_value)