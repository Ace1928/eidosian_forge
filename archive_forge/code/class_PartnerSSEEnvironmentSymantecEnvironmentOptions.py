from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PartnerSSEEnvironmentSymantecEnvironmentOptions(_messages.Message):
    """Fields that are applicable iff sse_service is SYMANTEC_CLOUD_SWG.

  Fields:
    apiEndpoint: Optional. URL to use for calling the Symantec Locations API.
  """
    apiEndpoint = _messages.StringField(1)