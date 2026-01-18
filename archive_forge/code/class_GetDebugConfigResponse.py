from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GetDebugConfigResponse(_messages.Message):
    """Response to a get debug configuration request.

  Fields:
    config: The encoded debug configuration for the requested component.
  """
    config = _messages.StringField(1)