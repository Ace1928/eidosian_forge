from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
Waits for an operation to complete.

    Polls the Firewall Plus Operation service until the operation completes,
    fails, or max_wait_seconds elapses.

    Args:
      operation_ref: A Resource created by GetOperationRef describing the
        Operation.
      message: The message to display to the user while they wait.
      has_result: If True, the function will return the target of the operation
        (the Firewall Plus Endpoint) when it completes. If False, nothing will
        be returned (useful for Delete operations)
      max_wait: The time to wait for the operation to succeed before timing out.

    Returns:
      If has_result = True, an Endpoint entity.
      Otherwise, None.
    