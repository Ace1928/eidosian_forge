from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OrgpolicyOrganizationsCustomConstraintsPatchRequest(_messages.Message):
    """A OrgpolicyOrganizationsCustomConstraintsPatchRequest object.

  Fields:
    googleCloudOrgpolicyV2CustomConstraint: A
      GoogleCloudOrgpolicyV2CustomConstraint resource to be passed as the
      request body.
    name: Immutable. Name of the constraint. This is unique within the
      organization. Format of the name should be * `organizations/{organizatio
      n_id}/customConstraints/{custom_constraint_id}` Example:
      `organizations/123/customConstraints/custom.createOnlyE2TypeVms` The max
      length is 70 characters and the minimum length is 1. Note that the
      prefix `organizations/{organization_id}/customConstraints/` is not
      counted.
  """
    googleCloudOrgpolicyV2CustomConstraint = _messages.MessageField('GoogleCloudOrgpolicyV2CustomConstraint', 1)
    name = _messages.StringField(2, required=True)