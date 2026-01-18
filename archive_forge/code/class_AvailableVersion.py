from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AvailableVersion(_messages.Message):
    """Deprecated.

  Fields:
    reason: Reason for availability.
    version: Kubernetes version.
  """
    reason = _messages.StringField(1)
    version = _messages.StringField(2)