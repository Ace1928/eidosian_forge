from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1BatchUpdateSecurityIncidentsRequest(_messages.Message):
    """Request for BatchUpdateSecurityIncident.

  Fields:
    requests: Optional. Required. The request message specifying the resources
      to update. A maximum of 1000 can be modified in a batch.
  """
    requests = _messages.MessageField('GoogleCloudApigeeV1UpdateSecurityIncidentRequest', 1, repeated=True)