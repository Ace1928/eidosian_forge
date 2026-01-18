from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MostRecentStartPosition(_messages.Message):
    """CDC strategy to start replicating from the most recent position in the
  source.
  """