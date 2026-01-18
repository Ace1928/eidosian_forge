from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QuorumType(_messages.Message):
    """Information about the database quorum type. this applies only for dual
  region instance configs.

  Fields:
    dualRegion: Dual region quorum type.
    singleRegion: Single region quorum type.
  """
    dualRegion = _messages.MessageField('DualRegionQuorum', 1)
    singleRegion = _messages.MessageField('SingleRegionQuorum', 2)