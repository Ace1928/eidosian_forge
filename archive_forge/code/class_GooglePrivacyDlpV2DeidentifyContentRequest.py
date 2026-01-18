from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DeidentifyContentRequest(_messages.Message):
    """Request to de-identify a ContentItem.

  Fields:
    deidentifyConfig: Configuration for the de-identification of the content
      item. Items specified here will override the template referenced by the
      deidentify_template_name argument.
    deidentifyTemplateName: Template to use. Any configuration directly
      specified in deidentify_config will override those set in the template.
      Singular fields that are set in this request will replace their
      corresponding fields in the template. Repeated fields are appended.
      Singular sub-messages and groups are recursively merged.
    inspectConfig: Configuration for the inspector. Items specified here will
      override the template referenced by the inspect_template_name argument.
    inspectTemplateName: Template to use. Any configuration directly specified
      in inspect_config will override those set in the template. Singular
      fields that are set in this request will replace their corresponding
      fields in the template. Repeated fields are appended. Singular sub-
      messages and groups are recursively merged.
    item: The item to de-identify. Will be treated as text. This value must be
      of type Table if your deidentify_config is a RecordTransformations
      object.
    locationId: Deprecated. This field has no effect.
  """
    deidentifyConfig = _messages.MessageField('GooglePrivacyDlpV2DeidentifyConfig', 1)
    deidentifyTemplateName = _messages.StringField(2)
    inspectConfig = _messages.MessageField('GooglePrivacyDlpV2InspectConfig', 3)
    inspectTemplateName = _messages.StringField(4)
    item = _messages.MessageField('GooglePrivacyDlpV2ContentItem', 5)
    locationId = _messages.StringField(6)