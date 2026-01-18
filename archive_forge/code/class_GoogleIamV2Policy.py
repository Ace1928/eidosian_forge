from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamV2Policy(_messages.Message):
    """Data for an IAM policy.

  Messages:
    AnnotationsValue: A key-value map to store arbitrary metadata for the
      `Policy`. Keys can be up to 63 characters. Values can be up to 255
      characters.

  Fields:
    annotations: A key-value map to store arbitrary metadata for the `Policy`.
      Keys can be up to 63 characters. Values can be up to 255 characters.
    createTime: Output only. The time when the `Policy` was created.
    deleteTime: Output only. The time when the `Policy` was deleted. Empty if
      the policy is not deleted.
    displayName: A user-specified description of the `Policy`. This value can
      be up to 63 characters.
    etag: An opaque tag that identifies the current version of the `Policy`.
      IAM uses this value to help manage concurrent updates, so they do not
      cause one update to be overwritten by another. If this field is present
      in a CreatePolicyRequest, the value is ignored.
    kind: Output only. The kind of the `Policy`. Always contains the value
      `DenyPolicy`.
    managingAuthority: Immutable. Specifies that this policy is managed by an
      authority and can only be modified by that authority. Usage is
      restricted.
    name: Immutable. The resource name of the `Policy`, which must be unique.
      Format: `policies/{attachment_point}/denypolicies/{policy_id}` The
      attachment point is identified by its URL-encoded full resource name,
      which means that the forward-slash character, `/`, must be written as
      `%2F`. For example,
      `policies/cloudresourcemanager.googleapis.com%2Fprojects%2Fmy-
      project/denypolicies/my-deny-policy`. For organizations and folders, use
      the numeric ID in the full resource name. For projects, requests can use
      the alphanumeric or the numeric ID. Responses always contain the numeric
      ID.
    rules: A list of rules that specify the behavior of the `Policy`. All of
      the rules should be of the `kind` specified in the `Policy`.
    uid: Immutable. The globally unique ID of the `Policy`. Assigned
      automatically when the `Policy` is created.
    updateTime: Output only. The time when the `Policy` was last updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """A key-value map to store arbitrary metadata for the `Policy`. Keys can
    be up to 63 characters. Values can be up to 255 characters.

    Messages:
      AdditionalProperty: An additional property for a AnnotationsValue
        object.

    Fields:
      additionalProperties: Additional properties of type AnnotationsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AnnotationsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    annotations = _messages.MessageField('AnnotationsValue', 1)
    createTime = _messages.StringField(2)
    deleteTime = _messages.StringField(3)
    displayName = _messages.StringField(4)
    etag = _messages.StringField(5)
    kind = _messages.StringField(6)
    managingAuthority = _messages.StringField(7)
    name = _messages.StringField(8)
    rules = _messages.MessageField('GoogleIamV2PolicyRule', 9, repeated=True)
    uid = _messages.StringField(10)
    updateTime = _messages.StringField(11)