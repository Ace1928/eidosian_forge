from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityPolicyAdaptiveProtectionConfigAutoDeployConfig(_messages.Message):
    """Configuration options for Adaptive Protection auto-deploy feature.

  Fields:
    confidenceThreshold: A number attribute.
    expirationSec: A integer attribute.
    impactedBaselineThreshold: A number attribute.
    loadThreshold: A number attribute.
  """
    confidenceThreshold = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    expirationSec = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    impactedBaselineThreshold = _messages.FloatField(3, variant=_messages.Variant.FLOAT)
    loadThreshold = _messages.FloatField(4, variant=_messages.Variant.FLOAT)