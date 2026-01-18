from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplaneV2FeatureSpec(_messages.Message):
    """**Dataplane V2**: Spec

  Fields:
    enableEncryption: Enable dataplane-v2 based encryption for multiple
      clusters.
  """
    enableEncryption = _messages.BooleanField(1)