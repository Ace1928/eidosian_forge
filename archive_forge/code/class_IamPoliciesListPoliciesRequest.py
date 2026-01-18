from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamPoliciesListPoliciesRequest(_messages.Message):
    """A IamPoliciesListPoliciesRequest object.

  Fields:
    pageSize: The maximum number of policies to return. IAM ignores this value
      and uses the value 1000.
    pageToken: A page token received in a ListPoliciesResponse. Provide this
      token to retrieve the next page.
    parent: Required. The resource that the policy is attached to, along with
      the kind of policy to list. Format:
      `policies/{attachment_point}/denypolicies` The attachment point is
      identified by its URL-encoded full resource name, which means that the
      forward-slash character, `/`, must be written as `%2F`. For example,
      `policies/cloudresourcemanager.googleapis.com%2Fprojects%2Fmy-
      project/denypolicies`. For organizations and folders, use the numeric ID
      in the full resource name. For projects, you can use the alphanumeric or
      the numeric ID.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)