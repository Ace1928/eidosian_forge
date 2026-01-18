from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import time
import enum
from googlecloudsdk.api_lib.logging import common as logging_common
from googlecloudsdk.core import log
from googlecloudsdk.core.util import times
def GetLogs(self):
    """Retrieves a batch of logs.

    After we fetch the logs, we ensure that none of the logs have been seen
    before.  Along the way, we update the most recent timestamp.

    Returns:
      A list of valid log entries.
    """
    utcnow = datetime.datetime.utcnow()
    lower_filter = self.log_position.GetFilterLowerBound()
    upper_filter = self.log_position.GetFilterUpperBound(utcnow)
    new_filter = self.base_filters + [lower_filter, upper_filter]
    entries = logging_common.FetchLogs(log_filter=' AND '.join(new_filter), order_by='ASC', limit=self.LOG_BATCH_SIZE)
    return [entry for entry in entries if self.log_position.Update(entry.timestamp, entry.insertId)]