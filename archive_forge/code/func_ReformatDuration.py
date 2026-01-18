from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import time
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import exceptions
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
def ReformatDuration(duration):
    """Reformat a Duration arg to work around ApiTools non-support of that type.

  Duration args are normally converted to an int in seconds (e.g. --timeout 5m
  becomes args.timeout with int value 300). Duration proto fields are converted
  to type string during discovery doc creation, so we have to convert the int
  back into a string-formatted Duration (i.e. append an 's') before
  passing it to the Testing Service.

  Args:
    duration: {int} the number of seconds in the time duration.

  Returns:
    String representation of the Duration with units of seconds.
  """
    return '{secs}s'.format(secs=duration)