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
def _ValidateListOfStringToStringDicts(arg_internal_name, arg_value):
    """Validates that an argument is a list of dicts of key=value string pairs."""
    if not isinstance(arg_value, list):
        raise test_exceptions.InvalidArgException(arg_internal_name, 'is not a list of maps of key-value pairs.')
    new_list = []
    for a_dict in arg_value:
        if not isinstance(a_dict, dict):
            raise test_exceptions.InvalidArgException(arg_internal_name, 'Each list item must be a map of key-value string pairs.')
        new_dict = {}
        for key, value in a_dict.items():
            new_dict[str(key)] = _ValidateString(key, value)
        new_list.append(new_dict)
    return new_list