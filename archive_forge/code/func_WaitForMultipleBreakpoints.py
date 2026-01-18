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
def WaitForMultipleBreakpoints(self, ids, wait_all=False, timeout=None):
    """Waits for one or more breakpoints to complete.

    Args:
      ids: A list of breakpoint IDs.
      wait_all: If True, wait for all breakpoints to complete. Otherwise, wait
        for any breakpoint to complete.
      timeout: The number of seconds to wait for completion.
    Returns:
      The completed Breakpoint messages, in the order requested. If wait_all was
      specified and the timeout was reached, the result will still comprise the
      completed Breakpoints.
    """
    waiter = _BreakpointWaiter(wait_all, timeout)
    for i in ids:
        waiter.AddTarget(self, i)
    results = waiter.Wait()
    return [results[i] for i in ids if i in results]