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
class AppInfoSummary(validation.Validated):
    """This class contains only basic summary information about an app.

  This class is used to pass back information about the newly created app to
  users after a new version has been created.
  """
    ATTRIBUTES = {APPLICATION: APPLICATION_RE_STRING, MAJOR_VERSION: MODULE_VERSION_ID_RE_STRING, MINOR_VERSION: validation.TYPE_LONG}