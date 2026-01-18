from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1SerializedTaxonomy(_messages.Message):
    """A nested protocol buffer that represents a taxonomy and the hierarchy of
  its policy tags. Used for taxonomy replacement, import, and export.

  Enums:
    ActivatedPolicyTypesValueListEntryValuesEnum:

  Fields:
    activatedPolicyTypes: A list of policy types that are activated per
      taxonomy.
    description: Description of the serialized taxonomy. At most 2000 bytes
      when encoded in UTF-8. If not set, defaults to an empty description.
    displayName: Required. Display name of the taxonomy. At most 200 bytes
      when encoded in UTF-8.
    policyTags: Top level policy tags associated with the taxonomy, if any.
  """

    class ActivatedPolicyTypesValueListEntryValuesEnum(_messages.Enum):
        """ActivatedPolicyTypesValueListEntryValuesEnum enum type.

    Values:
      POLICY_TYPE_UNSPECIFIED: Unspecified policy type.
      FINE_GRAINED_ACCESS_CONTROL: Fine-grained access control policy that
        enables access control on tagged sub-resources.
    """
        POLICY_TYPE_UNSPECIFIED = 0
        FINE_GRAINED_ACCESS_CONTROL = 1
    activatedPolicyTypes = _messages.EnumField('ActivatedPolicyTypesValueListEntryValuesEnum', 1, repeated=True)
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)
    policyTags = _messages.MessageField('GoogleCloudDatacatalogV1SerializedPolicyTag', 4, repeated=True)