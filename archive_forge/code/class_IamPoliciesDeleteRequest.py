from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamPoliciesDeleteRequest(_messages.Message):
    """A IamPoliciesDeleteRequest object.

  Fields:
    etag: Optional. The expected `etag` of the policy to delete. If the value
      does not match the value that is stored in IAM, the request fails with a
      `409` error code and `ABORTED` status. If you omit this field, the
      policy is deleted regardless of its current `etag`.
    name: Required. The resource name of the policy to delete. Format:
      `policies/{attachment_point}/denypolicies/{policy_id}` Use the URL-
      encoded full resource name, which means that the forward-slash
      character, `/`, must be written as `%2F`. For example,
      `policies/cloudresourcemanager.googleapis.com%2Fprojects%2Fmy-
      project/denypolicies/my-policy`. For organizations and folders, use the
      numeric ID in the full resource name. For projects, you can use the
      alphanumeric or the numeric ID.
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2, required=True)