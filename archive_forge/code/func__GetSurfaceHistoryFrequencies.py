from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
import re
from googlecloudsdk.command_lib.static_completion import lookup
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
import six
def _GetSurfaceHistoryFrequencies(logs_dir):
    """Load the last 100 surfaces user used today from local command history.

  Args:
    logs_dir: str, the path to today's logs directory

  Returns:
    dict mapping surfaces to normalized frequencies.
  """
    surfaces_count = collections.defaultdict(int)
    if not logs_dir:
        return surfaces_count
    total = 0
    last_100_invocations = sorted(os.listdir(logs_dir), reverse=True)[:100]
    for filename in last_100_invocations:
        file_path = os.path.join(logs_dir, filename)
        with files.FileReader(file_path) as log_file:
            for line in log_file:
                match = re.search(log.USED_SURFACE_PATTERN, line)
                if match:
                    surface = match.group(1)
                    total += 1
                    surfaces_count[surface] += 1
    return {surface: count / total for surface, count in six.iteritems(surfaces_count)}