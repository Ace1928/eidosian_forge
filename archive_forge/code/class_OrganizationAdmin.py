from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OrganizationAdmin(_messages.Message):
    """List of admins that will be granted with GCP IAM role:
  roles/resourcemanager.organizationAdmin

  Fields:
    member: Required. Valid IAM principles. See member field under
      https://cloud.google.com/iam/docs/reference/sts/rest/v1/Binding
  """
    member = _messages.StringField(1)