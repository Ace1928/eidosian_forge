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
def GetAllRuntimes():
    """Returns the list of all valid runtimes.

  This list can include third-party runtimes as well as canned runtimes.

  Returns:
    Tuple of strings.
  """
    return _all_runtimes