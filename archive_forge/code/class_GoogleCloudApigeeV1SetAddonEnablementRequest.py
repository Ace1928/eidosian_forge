from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1SetAddonEnablementRequest(_messages.Message):
    """Request for SetAddonEnablement.

  Fields:
    analyticsEnabled: If the Analytics should be enabled in the environment.
    apiSecurityEnabled: If the API Security should be enabled in the
      environment.
  """
    analyticsEnabled = _messages.BooleanField(1)
    apiSecurityEnabled = _messages.BooleanField(2)