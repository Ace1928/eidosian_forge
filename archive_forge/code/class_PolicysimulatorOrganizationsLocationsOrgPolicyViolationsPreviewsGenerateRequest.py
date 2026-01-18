from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicysimulatorOrganizationsLocationsOrgPolicyViolationsPreviewsGenerateRequest(_messages.Message):
    """A PolicysimulatorOrganizationsLocationsOrgPolicyViolationsPreviewsGenera
  teRequest object.

  Fields:
    googleCloudPolicysimulatorV1betaOrgPolicyViolationsPreview: A
      GoogleCloudPolicysimulatorV1betaOrgPolicyViolationsPreview resource to
      be passed as the request body.
    parent: Required. The organization under which this
      OrgPolicyViolationsPreview will be created. Example: `organizations/my-
      example-org/locations/global`
  """
    googleCloudPolicysimulatorV1betaOrgPolicyViolationsPreview = _messages.MessageField('GoogleCloudPolicysimulatorV1betaOrgPolicyViolationsPreview', 1)
    parent = _messages.StringField(2, required=True)