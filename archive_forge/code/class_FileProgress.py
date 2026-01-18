from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import datetime
import enum
import threading
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import manifest_util
from googlecloudsdk.command_lib.storage import metrics_util
from googlecloudsdk.command_lib.storage import thread_messages
from googlecloudsdk.core import log
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import scaled_integer
import six
class FileProgress:
    """Holds progress information for file being copied.

  Attributes:
    component_progress (dict<int,int>): Records bytes copied per component. If
      not multi-component copy (e.g. "sliced download"), there will only be one
      component.
    start_time (datetime|None): Needed if writing file copy results to manifest.
    total_bytes_copied (int|None): Sum of bytes copied for each component.
      Needed because components are popped when completed, but we don't want to
      lose info on them if writing to the manifest.
  """

    def __init__(self, component_count, start_time=None, total_bytes_copied=None):
        self.component_progress = {i: 0 for i in range(component_count)}
        self.start_time = start_time
        self.total_bytes_copied = total_bytes_copied