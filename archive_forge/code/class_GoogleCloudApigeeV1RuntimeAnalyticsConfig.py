from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1RuntimeAnalyticsConfig(_messages.Message):
    """Runtime configuration for the Analytics add-on.

  Fields:
    billingPipelineEnabled: If Runtime should send billing data to AX or not.
    enabled: If the Analytics is enabled or not.
  """
    billingPipelineEnabled = _messages.BooleanField(1)
    enabled = _messages.BooleanField(2)