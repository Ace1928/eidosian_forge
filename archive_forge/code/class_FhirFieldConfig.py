from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FhirFieldConfig(_messages.Message):
    """Specifies how to handle the de-identification of a FHIR store.

  Enums:
    ProfileTypeValueValuesEnum: Base profile type for handling FHIR fields.

  Fields:
    fieldMetadataList: Specifies FHIR paths to match and how to transform
      them. Any field that is not matched by a FieldMetadata `action` is
      passed through to the output dataset unmodified. All extensions will be
      processed according to keep_extensions. If a field can be matched by
      more than one FieldMetadata `action`, the first `action` option is
      applied. Overrides options and the union field `profile` in
      FhirFieldConfig.
    options: Specifies additional options, overriding the base ProfileType.
    profileType: Base profile type for handling FHIR fields.
  """

    class ProfileTypeValueValuesEnum(_messages.Enum):
        """Base profile type for handling FHIR fields.

    Values:
      PROFILE_TYPE_UNSPECIFIED: No profile provided. Same as `BASIC`.
      KEEP_ALL: Keep all fields.
      BASIC: Transforms known [HIPAA 18](https://www.hhs.gov/hipaa/for-
        professionals/privacy/special-topics/de-
        identification/index.html#standard) fields and cleans known
        unstructured text fields.
      CLEAN_ALL: Cleans all supported tags. Applies to types: Code, Date,
        DateTime, Decimal, HumanName, Id, LanguageCode, Markdown, Oid, String,
        Uri, Uuid, Xhtml.
    """
        PROFILE_TYPE_UNSPECIFIED = 0
        KEEP_ALL = 1
        BASIC = 2
        CLEAN_ALL = 3
    fieldMetadataList = _messages.MessageField('GoogleCloudHealthcareV1beta1DeidentifyFieldMetadata', 1, repeated=True)
    options = _messages.MessageField('GoogleCloudHealthcareV1beta1DeidentifyOptions', 2)
    profileType = _messages.EnumField('ProfileTypeValueValuesEnum', 3)