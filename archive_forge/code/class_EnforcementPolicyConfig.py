from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class EnforcementPolicyConfig(_messages.Message):
    """Continuous validation config for enforcement policy.

  Fields:
    enabled: Whether continuous validation is enabled for enforcement policy.
  """
    enabled = _messages.BooleanField(1)