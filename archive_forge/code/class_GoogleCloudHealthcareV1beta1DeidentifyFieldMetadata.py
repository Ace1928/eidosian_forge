from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudHealthcareV1beta1DeidentifyFieldMetadata(_messages.Message):
    """Specifies the FHIR paths to match and how to handle the de-
  identification of matching fields.

  Fields:
    characterMaskField: Replace the field's value with a masking character.
      Supported [types](https://www.hl7.org/fhir/datatypes.html): Code,
      Decimal, HumanName, Id, LanguageCode, Markdown, Oid, String, Uri, Uuid,
      Xhtml.
    cleanTextField: Inspect the field's text and transform sensitive text.
      Configure using TextConfig. Supported
      [types](https://www.hl7.org/fhir/datatypes.html): Code, Date, DateTime,
      Decimal, HumanName, Id, LanguageCode, Markdown, Oid, String, Uri, Uuid,
      Xhtml.
    cryptoHashField: Replace field value with a hash of that value. Supported
      [types](https://www.hl7.org/fhir/datatypes.html): Code, Decimal,
      HumanName, Id, LanguageCode, Markdown, Oid, String, Uri, Uuid, Xhtml.
    dateShiftField: Shift the date by a randomized number of days. See [date
      shifting](https://cloud.google.com/dlp/docs/concepts-date-shifting) for
      more information. Supported
      [types](https://www.hl7.org/fhir/datatypes.html): Date, DateTime.
    keepField: Keep the field unchanged.
    paths: List of paths to FHIR fields to redact. Each path is a period-
      separated list where each component is either a field name or FHIR
      [type](https://www.hl7.org/fhir/datatypes.html) name. All types begin
      with an upper case letter. For example, the resource field
      `Patient.Address.city`, which uses a
      [string](https://www.hl7.org/fhir/datatypes-
      definitions.html#Address.city) type, can be matched by
      `Patient.Address.String`. Partial matching is supported. For example,
      `Patient.Address.city` can be matched by `Address.city` (with `Patient`
      omitted). Partial matching and type matching can be combined, for
      example `Patient.Address.city` can be matched by `Address.String`. For
      "choice" types (those defined in the FHIR spec with the format
      `field[x]`), use two separate components. For example,
      `deceasedAge.unit` is matched by `Deceased.Age.unit`. The following
      types are supported: AdministrativeGenderCode, Base64Binary, Boolean,
      Code, Date, DateTime, Decimal, HumanName, Id, Instant, Integer,
      LanguageCode, Markdown, Oid, PositiveInt, String, UnsignedInt, Uri,
      Uuid, Xhtml. The sub-type for HumanName (for example `HumanName.given`,
      `HumanName.family`) can be omitted.
    removeField: Remove the field.
  """
    characterMaskField = _messages.MessageField('CharacterMaskField', 1)
    cleanTextField = _messages.MessageField('CleanTextField', 2)
    cryptoHashField = _messages.MessageField('CryptoHashField', 3)
    dateShiftField = _messages.MessageField('DateShiftField', 4)
    keepField = _messages.MessageField('KeepField', 5)
    paths = _messages.StringField(6, repeated=True)
    removeField = _messages.MessageField('RemoveField', 7)