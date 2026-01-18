from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReverseReplicationDirectionRequest(_messages.Message):
    """ReverseReplicationDirectionRequest reverses direction of replication.
  Source becomes destination and destination becomes source.
  """