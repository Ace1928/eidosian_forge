from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StateFilterValueValuesEnum(_messages.Enum):
    """Filter for job state

    Values:
      done: Finished jobs
      pending: Pending jobs
      running: Running jobs
    """
    done = 0
    pending = 1
    running = 2