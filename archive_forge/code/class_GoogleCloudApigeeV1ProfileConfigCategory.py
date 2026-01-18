from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ProfileConfigCategory(_messages.Message):
    """Advanced API Security provides security profile that scores the
  following categories.

  Fields:
    abuse: Checks for abuse, which includes any requests sent to the API for
      purposes other than what it is intended for, such as high volumes of
      requests, data scraping, and abuse related to authorization.
    authorization: Checks to see if you have an authorization policy in place.
    cors: Checks to see if you have CORS policy in place.
    mediation: Checks to see if you have a mediation policy in place.
    mtls: Checks to see if you have configured mTLS for the target server.
    threat: Checks to see if you have a threat protection policy in place.
  """
    abuse = _messages.MessageField('GoogleCloudApigeeV1ProfileConfigAbuse', 1)
    authorization = _messages.MessageField('GoogleCloudApigeeV1ProfileConfigAuthorization', 2)
    cors = _messages.MessageField('GoogleCloudApigeeV1ProfileConfigCORS', 3)
    mediation = _messages.MessageField('GoogleCloudApigeeV1ProfileConfigMediation', 4)
    mtls = _messages.MessageField('GoogleCloudApigeeV1ProfileConfigMTLS', 5)
    threat = _messages.MessageField('GoogleCloudApigeeV1ProfileConfigThreat', 6)