from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AssuredworkloadsOrganizationsLocationsWorkloadsViolationsAcknowledgeRequest(_messages.Message):
    """A
  AssuredworkloadsOrganizationsLocationsWorkloadsViolationsAcknowledgeRequest
  object.

  Fields:
    googleCloudAssuredworkloadsV1AcknowledgeViolationRequest: A
      GoogleCloudAssuredworkloadsV1AcknowledgeViolationRequest resource to be
      passed as the request body.
    name: Required. The resource name of the Violation to acknowledge. Format:
      organizations/{organization}/locations/{location}/workloads/{workload}/v
      iolations/{violation}
  """
    googleCloudAssuredworkloadsV1AcknowledgeViolationRequest = _messages.MessageField('GoogleCloudAssuredworkloadsV1AcknowledgeViolationRequest', 1)
    name = _messages.StringField(2, required=True)