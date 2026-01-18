from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.alloydb import api_util
from googlecloudsdk.api_lib.util import waiter
Waits for the given google.longrunning.Operation to complete.

  Args:
    op_ref: The operation to poll.
    message: String to display for default progress_tracker.
    release_track: The API release track (e.g. ALPHA, BETA, etc.)
    creates_resource: Whether or not the operation creates a resource.

  Raises:
    apitools.base.py.HttpError: If the request returns an HTTP error.

  Returns:
    The Operation or the Resource the Operation is associated with.
  