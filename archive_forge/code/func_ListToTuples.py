from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import re
import string
import sys
import wsgiref.util
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.api import backendinfo
from googlecloudsdk.third_party.appengine._internal import six_subset
@classmethod
def ListToTuples(cls, builtins_list):
    """Converts a list of `BuiltinHandler` objects.

    Args:
      builtins_list: A list of `BuildinHandler` objects to convert to tuples.

    Returns:
      A list of `(name, status)` that is derived from the `BuiltinHandler`
      objects.
    """
    return [(b.builtin_name, getattr(b, b.builtin_name)) for b in builtins_list]