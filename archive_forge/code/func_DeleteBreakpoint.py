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
def DeleteBreakpoint(self, breakpoint_id):
    """Deletes a breakpoint.

    Args:
      breakpoint_id: A breakpoint ID.
    """
    request = self._debug_messages.ClouddebuggerDebuggerDebuggeesBreakpointsDeleteRequest(breakpointId=breakpoint_id, debuggeeId=self.target_id, clientVersion=self.CLIENT_VERSION)
    try:
        self._debug_client.debugger_debuggees_breakpoints.Delete(request)
    except apitools_exceptions.HttpError as error:
        raise errors.UnknownHttpError(error)