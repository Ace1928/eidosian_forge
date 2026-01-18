from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_attr
def TrackerUpdateFunction(self, tracker, poll_result, status):
    """Custom tracker function which gets called after every tick.

    This gets called whenever progress tracker gets a tick. However we want to
    stream remote output to users instead of showing a progress tracker.

    Args:
      tracker: Progress tracker instance. Not being used.
      poll_result: Result from Poll function.
      status: Status argument that is supposed to pass to the progress tracker
      instance. Not being used here as well.
    """
    self._CheckStreamer(poll_result)
    self._StreamOutput()