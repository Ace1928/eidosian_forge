from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1SerializedPolicyTag(_messages.Message):
    """A nested protocol buffer that represents a policy tag and all its
  descendants.

  Fields:
    childPolicyTags: Children of the policy tag, if any.
    description: Description of the serialized policy tag. At most 2000 bytes
      when encoded in UTF-8. If not set, defaults to an empty description.
    displayName: Required. Display name of the policy tag. At most 200 bytes
      when encoded in UTF-8.
    policyTag: Resource name of the policy tag. This field is ignored when
      calling `ImportTaxonomies`.
  """
    childPolicyTags = _messages.MessageField('GoogleCloudDatacatalogV1SerializedPolicyTag', 1, repeated=True)
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)
    policyTag = _messages.StringField(4)