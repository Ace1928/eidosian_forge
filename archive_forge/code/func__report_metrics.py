from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.tasks import task_util
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
def _report_metrics(self, total_bytes, time_delta, num_files):
    """Reports back all tracked events via report method.

    Args:
      total_bytes (int): Amount of data transferred in bytes.
      time_delta (int): Time elapsed during the transfer in seconds.
      num_files (int): Number of files processed
    """
    avg_speed = round(float(total_bytes) / float(time_delta))
    report(source_scheme=self._source_scheme, destination_scheme=self._destination_scheme, num_files=num_files, size=total_bytes, avg_speed=avg_speed, disk_io_time=self._calculate_disk_io())