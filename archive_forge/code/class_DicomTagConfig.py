from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DicomTagConfig(_messages.Message):
    """Specifies the parameters needed for the de-identification of DICOM
  stores.

  Enums:
    ProfileTypeValueValuesEnum: Base profile type for handling DICOM tags.

  Fields:
    actions: Specifies custom tag selections and `Actions` to apply to them.
      Overrides `options` and `profile`. Conflicting `Actions` are applied in
      the order given.
    options: Specifies additional options to apply, overriding the base
      `profile`.
    profileType: Base profile type for handling DICOM tags.
  """

    class ProfileTypeValueValuesEnum(_messages.Enum):
        """Base profile type for handling DICOM tags.

    Values:
      PROFILE_TYPE_UNSPECIFIED: No profile provided. Same as
        `ATTRIBUTE_CONFIDENTIALITY_BASIC_PROFILE`.
      MINIMAL_KEEP_LIST_PROFILE: Keep only the tags required to produce valid
        DICOM objects.
      ATTRIBUTE_CONFIDENTIALITY_BASIC_PROFILE: Remove tags based on DICOM
        Standard's [Attribute Confidentiality Basic Profile (DICOM Standard
        Edition 2018e)](http://dicom.nema.org/medical/dicom/2018e/output/chtml
        /part15/chapter_E.html).
      KEEP_ALL_PROFILE: Keep all tags.
      DEIDENTIFY_TAG_CONTENTS: Inspect tag contents and replace sensitive
        text. The process can be configured using the TextConfig. Applies to
        all tags with the following [Value Representations] (http://dicom.nema
        .org/medical/dicom/2018e/output/chtml/part05/sect_6.2.html#table_6.2-
        1): AE, LO, LT, PN, SH, ST, UC, UT, DA, DT, AS
    """
        PROFILE_TYPE_UNSPECIFIED = 0
        MINIMAL_KEEP_LIST_PROFILE = 1
        ATTRIBUTE_CONFIDENTIALITY_BASIC_PROFILE = 2
        KEEP_ALL_PROFILE = 3
        DEIDENTIFY_TAG_CONTENTS = 4
    actions = _messages.MessageField('Action', 1, repeated=True)
    options = _messages.MessageField('Options', 2)
    profileType = _messages.EnumField('ProfileTypeValueValuesEnum', 3)