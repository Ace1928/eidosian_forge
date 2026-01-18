from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import threading
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.debug import errors
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import retry
import six
from six.moves import urllib
def _BreakpointMatchesIdOrRegexp(breakpoint, ids, patterns):
    """Check if a breakpoint matches any of the given IDs or regexps.

  Args:
    breakpoint: Any _debug_messages.Breakpoint message object.
    ids: A set of strings to search for exact matches on breakpoint ID.
    patterns: A list of regular expressions to match against the file:line
      location of the breakpoint.
  Returns:
    True if the breakpoint matches any ID or pattern.
  """
    if breakpoint.id in ids:
        return True
    if not breakpoint.location:
        return False
    location = _FormatLocation(breakpoint.location)
    for p in patterns:
        if p.match(location):
            return True
    return False