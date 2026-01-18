from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityPolicyCloudArmorConfig(_messages.Message):
    """Configuration options for Cloud Armor.

  Fields:
    enableMl: If set to true, enables Cloud Armor Machine Learning.
  """
    enableMl = _messages.BooleanField(1)