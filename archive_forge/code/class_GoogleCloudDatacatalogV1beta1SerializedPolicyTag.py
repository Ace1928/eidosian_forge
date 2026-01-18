from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1beta1SerializedPolicyTag(_messages.Message):
    """Message representing one policy tag when exported as a nested proto.

  Fields:
    childPolicyTags: Children of the policy tag if any.
    description: Description of the serialized policy tag. The length of the
      description is limited to 2000 bytes when encoded in UTF-8. If not set,
      defaults to an empty description.
    displayName: Required. Display name of the policy tag. Max 200 bytes when
      encoded in UTF-8.
    policyTag: Resource name of the policy tag. This field will be ignored
      when calling ImportTaxonomies.
  """
    childPolicyTags = _messages.MessageField('GoogleCloudDatacatalogV1beta1SerializedPolicyTag', 1, repeated=True)
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)
    policyTag = _messages.StringField(4)