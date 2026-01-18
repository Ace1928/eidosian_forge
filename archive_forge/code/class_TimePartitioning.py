from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TimePartitioning(_messages.Message):
    """A TimePartitioning object.

  Fields:
    expirationMs: [Optional] Number of milliseconds for which to keep the
      storage for a partition.
    type: [Required] The only type supported is DAY, which will generate one
      partition per day based on data loading time.
  """
    expirationMs = _messages.IntegerField(1)
    type = _messages.StringField(2)