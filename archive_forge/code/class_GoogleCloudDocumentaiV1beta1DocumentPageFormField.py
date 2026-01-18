from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta1DocumentPageFormField(_messages.Message):
    """A form field detected on the page.

  Fields:
    correctedKeyText: Created for Labeling UI to export key text. If
      corrections were made to the text identified by the
      `field_name.text_anchor`, this field will contain the correction.
    correctedValueText: Created for Labeling UI to export value text. If
      corrections were made to the text identified by the
      `field_value.text_anchor`, this field will contain the correction.
    fieldName: Layout for the FormField name. e.g. `Address`, `Email`, `Grand
      total`, `Phone number`, etc.
    fieldValue: Layout for the FormField value.
    nameDetectedLanguages: A list of detected languages for name together with
      confidence.
    provenance: The history of this annotation.
    valueDetectedLanguages: A list of detected languages for value together
      with confidence.
    valueType: If the value is non-textual, this field represents the type.
      Current valid values are: - blank (this indicates the `field_value` is
      normal text) - `unfilled_checkbox` - `filled_checkbox`
  """
    correctedKeyText = _messages.StringField(1)
    correctedValueText = _messages.StringField(2)
    fieldName = _messages.MessageField('GoogleCloudDocumentaiV1beta1DocumentPageLayout', 3)
    fieldValue = _messages.MessageField('GoogleCloudDocumentaiV1beta1DocumentPageLayout', 4)
    nameDetectedLanguages = _messages.MessageField('GoogleCloudDocumentaiV1beta1DocumentPageDetectedLanguage', 5, repeated=True)
    provenance = _messages.MessageField('GoogleCloudDocumentaiV1beta1DocumentProvenance', 6)
    valueDetectedLanguages = _messages.MessageField('GoogleCloudDocumentaiV1beta1DocumentPageDetectedLanguage', 7, repeated=True)
    valueType = _messages.StringField(8)