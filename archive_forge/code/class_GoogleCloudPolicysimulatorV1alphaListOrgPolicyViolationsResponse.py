from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicysimulatorV1alphaListOrgPolicyViolationsResponse(_messages.Message):
    """ListOrgPolicyViolationsResponse is the response message for
  OrgPolicyViolationsPreviewService.ListOrgPolicyViolations

  Fields:
    nextPageToken: A token that you can use to retrieve the next page of
      results. If this field is omitted, there are no subsequent pages.
    orgPolicyViolations: The list of OrgPolicyViolations
  """
    nextPageToken = _messages.StringField(1)
    orgPolicyViolations = _messages.MessageField('GoogleCloudPolicysimulatorV1alphaOrgPolicyViolation', 2, repeated=True)