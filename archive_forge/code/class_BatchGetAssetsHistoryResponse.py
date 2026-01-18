from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchGetAssetsHistoryResponse(_messages.Message):
    """Batch get assets history response.

  Fields:
    assets: A list of assets with valid time windows.
  """
    assets = _messages.MessageField('TemporalAsset', 1, repeated=True)