from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StopBackfillJobRequest(_messages.Message):
    """Request for manually stopping a running backfill job for a specific
  stream object.
  """