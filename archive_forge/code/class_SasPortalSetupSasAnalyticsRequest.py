from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalSetupSasAnalyticsRequest(_messages.Message):
    """Request for the SetupSasAnalytics rpc.

  Fields:
    userId: Optional. User id to setup analytics for, if not provided the user
      id associated with the project is used. optional
  """
    userId = _messages.StringField(1)