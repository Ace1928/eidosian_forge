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
def ValidateHandlers(handlers, is_include_file=False):
    """Validates a list of handler (`URLMap`) objects.

  Args:
    handlers: A list of a handler (`URLMap`) objects.
    is_include_file: If this argument is set to `True`, the handlers that are
        added as part of the `includes` directive are validated.
  """
    if not handlers:
        return
    for handler in handlers:
        handler.FixSecureDefaults()
        handler.WarnReservedURLs()
        if not is_include_file:
            handler.ErrorOnPositionForAppInfo()