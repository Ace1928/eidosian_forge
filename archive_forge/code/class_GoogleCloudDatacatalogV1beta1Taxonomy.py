from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1beta1Taxonomy(_messages.Message):
    """A taxonomy is a collection of policy tags that classify data along a
  common axis. For instance a data *sensitivity* taxonomy could contain policy
  tags denoting PII such as age, zipcode, and SSN. A data *origin* taxonomy
  could contain policy tags to distinguish user data, employee data, partner
  data, public data.

  Enums:
    ActivatedPolicyTypesValueListEntryValuesEnum:

  Fields:
    activatedPolicyTypes: Optional. A list of policy types that are activated
      for this taxonomy. If not set, defaults to an empty list.
    description: Optional. Description of this taxonomy. It must: contain only
      unicode characters, tabs, newlines, carriage returns and page breaks;
      and be at most 2000 bytes long when encoded in UTF-8. If not set,
      defaults to an empty description.
    displayName: Required. User defined name of this taxonomy. It must:
      contain only unicode letters, numbers, underscores, dashes and spaces;
      not start or end with spaces; and be at most 200 bytes long when encoded
      in UTF-8. The taxonomy display name must be unique within an
      organization.
    name: Identifier. Resource name of this taxonomy, whose format is:
      "projects/{project_number}/locations/{location_id}/taxonomies/{id}".
    policyTagCount: Output only. Number of policy tags contained in this
      taxonomy.
    service: Output only. Identity of the service which owns the Taxonomy.
      This field is only populated when the taxonomy is created by a Google
      Cloud service. Currently only 'DATAPLEX' is supported.
    taxonomyTimestamps: Output only. Timestamps about this taxonomy. Only
      create_time and update_time are used.
  """

    class ActivatedPolicyTypesValueListEntryValuesEnum(_messages.Enum):
        """ActivatedPolicyTypesValueListEntryValuesEnum enum type.

    Values:
      POLICY_TYPE_UNSPECIFIED: Unspecified policy type.
      FINE_GRAINED_ACCESS_CONTROL: Fine grained access control policy, which
        enables access control on tagged resources.
    """
        POLICY_TYPE_UNSPECIFIED = 0
        FINE_GRAINED_ACCESS_CONTROL = 1
    activatedPolicyTypes = _messages.EnumField('ActivatedPolicyTypesValueListEntryValuesEnum', 1, repeated=True)
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)
    name = _messages.StringField(4)
    policyTagCount = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    service = _messages.MessageField('GoogleCloudDatacatalogV1beta1TaxonomyService', 6)
    taxonomyTimestamps = _messages.MessageField('GoogleCloudDatacatalogV1beta1SystemTimestamps', 7)