from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StopRuntimeRequest(_messages.Message):
    """Request for stopping a Managed Notebook Runtime.

  Fields:
    requestId: Idempotent request UUID.
  """
    requestId = _messages.StringField(1)