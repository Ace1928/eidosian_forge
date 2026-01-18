from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1BatchUpdateSecurityIncidentsResponse(_messages.Message):
    """Response for BatchUpdateSecurityIncident.

  Fields:
    securityIncidents: Output only. Updated security incidents
  """
    securityIncidents = _messages.MessageField('GoogleCloudApigeeV1SecurityIncident', 1, repeated=True)