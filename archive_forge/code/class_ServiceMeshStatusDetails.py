from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceMeshStatusDetails(_messages.Message):
    """Structured and human-readable details for a status.

  Fields:
    code: A machine-readable code that further describes a broad status.
    details: Human-readable explanation of code.
  """
    code = _messages.StringField(1)
    details = _messages.StringField(2)