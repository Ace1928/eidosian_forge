from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_attr
def _StreamOutput(self):
    if self.driver_log_streamer and self.driver_log_streamer.open:
        self.driver_log_streamer.ReadIntoWritable(log.err)