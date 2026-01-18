from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import os
import posixpath
import sys
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
from googlecloudsdk.third_party.appengine.api import client_deployinfo
import six
from six.moves import urllib
def GenerateVersionId(datetime_getter=datetime.datetime.now):
    """Generates a version id based off the current time.

  Args:
    datetime_getter: A function that returns a datetime.datetime instance.

  Returns:
    A version string based.
  """
    return datetime_getter().isoformat().lower().replace('-', '').replace(':', '')[:15]