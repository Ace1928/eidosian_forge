from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RawPayload(_messages.Message):
    """Message for storing a RAW ConfigType Config resource.

  Fields:
    data: Required. User provided content of a ConfigVersion. It can hold
      references to Secret Manager SecretVersion resources.
  """
    data = _messages.BytesField(1)