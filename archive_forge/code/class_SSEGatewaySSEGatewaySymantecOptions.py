from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SSEGatewaySSEGatewaySymantecOptions(_messages.Message):
    """Fields specific to SSEGWs connecting to Symantec Cloud SWG.

  Fields:
    symantecLocationName: Immutable. Name to be used for when creating a
      Location on the customer's behalf in Symantec's Location API. Required
      iff sse_realm uses SYMANTEC_CLOUD_SWG. Not to be confused with GCP
      locations.
    symantecSite: Immutable. Symantec data center identifier that this SSEGW
      will connect to. Required iff sse_realm uses SYMANTEC_CLOUD_SWG.
  """
    symantecLocationName = _messages.StringField(1)
    symantecSite = _messages.StringField(2)