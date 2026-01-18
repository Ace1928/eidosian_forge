from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterServiceperimeterV3alphaAccessPolicyExplanation(_messages.Message):
    """Explanation of an access policy NextTAG: 5

  Fields:
    accessLevelDetailedExplanations: Detailed explanations of access levels
      from the Access Level Troubleshooter Frontend Service
    accessPolicy: The full resource name of an access policy Format:
      `accessPolicies/{access_policy}`
    servicePerimeterExplanations: The explanations for the service perimeters
      in order
    servicePerimeters: The service perimeter definitions
  """
    accessLevelDetailedExplanations = _messages.MessageField('IdentityCaaIntelFrontendAccessLevelExplanation', 1, repeated=True)
    accessPolicy = _messages.StringField(2)
    servicePerimeterExplanations = _messages.MessageField('GoogleCloudPolicytroubleshooterServiceperimeterV3alphaServicePerimeterExplanation', 3, repeated=True)
    servicePerimeters = _messages.MessageField('GoogleIdentityAccesscontextmanagerV1ServicePerimeter', 4, repeated=True)