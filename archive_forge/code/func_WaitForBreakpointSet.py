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
def WaitForBreakpointSet(self, breakpoint_id, original_location, timeout=None, retry_ms=500):
    """Waits for a breakpoint to be set by at least one agent.

      Breakpoint set can be detected in two ways: it can be completed, or the
      location may change if the breakpoint could not be set at the specified
      location. A breakpoint may also be set without any change being reported
      to the server, in which case this function will wait until the timeout
      is reached.
    Args:
      breakpoint_id: A breakpoint ID.
      original_location: string, the user-specified breakpoint location. If a
        response has a different location, the function will return immediately.
      timeout: The number of seconds to wait for completion.
      retry_ms: Milliseconds to wait betweeen retries.
    Returns:
      The Breakpoint message, or None if the breakpoint did not get set before
      the timeout.
    """

    def MovedOrFinal(r):
        return r.breakpoint.isFinalState or (original_location and original_location != _FormatLocation(r.breakpoint.location))
    try:
        return self.WaitForBreakpoint(breakpoint_id=breakpoint_id, timeout=timeout, retry_ms=retry_ms, completion_test=MovedOrFinal)
    except apitools_exceptions.HttpError as error:
        raise errors.UnknownHttpError(error)