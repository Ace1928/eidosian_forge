from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import math
import re
import six
def CalculateThroughput(total_bytes_transferred, total_elapsed_time):
    """Calculates throughput and checks for a small total_elapsed_time.

  Args:
    total_bytes_transferred: Total bytes transferred in a period of time.
    total_elapsed_time: The amount of time elapsed in seconds.

  Returns:
    The throughput as a float.
  """
    if total_elapsed_time < 0.01:
        total_elapsed_time = 0.01
    return float(total_bytes_transferred) / float(total_elapsed_time)