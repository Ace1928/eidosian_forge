from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1SecuritySettings(_messages.Message):
    """SecuritySettings reflects the current state of the SecuritySettings
  feature.

  Fields:
    mlRetrainingFeedbackEnabled: Optional. If true the user consents to the
      use of ML models for Abuse detection.
    name: Identifier. Full resource name is always
      `organizations/{org}/securitySettings`.
  """
    mlRetrainingFeedbackEnabled = _messages.BooleanField(1)
    name = _messages.StringField(2)