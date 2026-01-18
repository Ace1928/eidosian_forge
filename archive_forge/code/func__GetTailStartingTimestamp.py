from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import time
import enum
from googlecloudsdk.api_lib.logging import common as logging_common
from googlecloudsdk.core import log
from googlecloudsdk.core.util import times
def _GetTailStartingTimestamp(filters, offset=None):
    """Returns the starting timestamp to start streaming logs from.

  Args:
    filters: [str], existing filters, should not contain timestamp constraints.
    offset: int, how many entries ago we should pick the starting timestamp.
      If not provided, unix time zero will be returned.

  Returns:
    str, A timestamp that can be used as lower bound or None if no lower bound
      is necessary.
  """
    if not offset:
        return None
    entries = list(logging_common.FetchLogs(log_filter=' AND '.join(filters), order_by='DESC', limit=offset))
    if len(entries) < offset:
        return None
    return list(entries)[-1].timestamp