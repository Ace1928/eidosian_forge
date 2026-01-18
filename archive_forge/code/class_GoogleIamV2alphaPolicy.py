from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamV2alphaPolicy(_messages.Message):
    """Represents policy data.

  Fields:
    createTime: Output only. The time when the `Policy` was created.
    deleteTime: Output only. The time when the `Policy` was deleted. Empty if
      the policy is not deleted.
    displayName: A user-specified opaque description of the `Policy`. Must be
      less than or equal to 63 characters.
    etag: An opaque tag indicating the current version of the `Policy`, used
      for concurrency control. When the `Policy` is returned from `GetPolicy`
      request, this `etag` indicates the version of the current `Policy` to
      use when executing a read-modify-write loop. When the `Policy` is used
      in a `UpdatePolicy` method, use the `etag` value that was returned from
      a `GetPolicy` request as part of a read-modify-write loop for
      concurrency control. This field is ignored if used in a `CreatePolicy`
      request.
    kind: Output only. The kind of the `Policy`. This is a read only field
      derived from the policy name.
    name: Immutable. The resource name of the `Policy`. Takes the form:
      `policies/{attachment-point}/{kind-plural}/{policy-id}` The attachment
      point is identified by its URL encoded full resource name, which means
      that the forward-slash character, '/', must be written as %2F. For
      example, `policies/cloudresourcemanager.googleapis.com%2Fprojects%2F123/
      denypolicies/a-deny-policy`.
    rules: List of rules that specify the behavior of the `Policy`. The list
      contains a single kind of rules, that matches the kind specified in the
      policy name.
    uid: Immutable. The globally unique ID of the `Policy`. This is a read
      only field assigned on policy creation.
    updateTime: Output only. The time when the `Policy` was last updated.
  """
    createTime = _messages.StringField(1)
    deleteTime = _messages.StringField(2)
    displayName = _messages.StringField(3)
    etag = _messages.StringField(4)
    kind = _messages.StringField(5)
    name = _messages.StringField(6)
    rules = _messages.MessageField('GoogleIamV2alphaPolicyRule', 7, repeated=True)
    uid = _messages.StringField(8)
    updateTime = _messages.StringField(9)