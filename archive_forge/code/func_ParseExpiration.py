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
def ParseExpiration(expiration):
    """Parses an expiration delta string.

  Args:
    expiration: String that matches `_DELTA_REGEX`.

  Returns:
    Time delta in seconds.
  """
    delta = 0
    for match in re.finditer(_DELTA_REGEX, expiration):
        amount = int(match.group(1))
        units = _EXPIRATION_CONVERSIONS.get(match.group(2).lower(), 1)
        delta += amount * units
    return delta