from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ServingSpec(_messages.Message):
    """ServingSpec defines the desired state of Serving

  Fields:
    enabled: A boolean attribute.
  """
    enabled = _messages.BooleanField(1)