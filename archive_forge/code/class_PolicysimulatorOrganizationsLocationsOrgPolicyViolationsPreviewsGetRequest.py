from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicysimulatorOrganizationsLocationsOrgPolicyViolationsPreviewsGetRequest(_messages.Message):
    """A
  PolicysimulatorOrganizationsLocationsOrgPolicyViolationsPreviewsGetRequest
  object.

  Fields:
    name: Required. The name of the OrgPolicyViolationsPreview to get.
  """
    name = _messages.StringField(1, required=True)