from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2ReidentifyContentRequest(_messages.Message):
    """Request to re-identify an item.

  Fields:
    inspectConfig: Configuration for the inspector.
    inspectTemplateName: Template to use. Any configuration directly specified
      in `inspect_config` will override those set in the template. Singular
      fields that are set in this request will replace their corresponding
      fields in the template. Repeated fields are appended. Singular sub-
      messages and groups are recursively merged.
    item: The item to re-identify. Will be treated as text.
    locationId: Deprecated. This field has no effect.
    reidentifyConfig: Configuration for the re-identification of the content
      item. This field shares the same proto message type that is used for de-
      identification, however its usage here is for the reversal of the
      previous de-identification. Re-identification is performed by examining
      the transformations used to de-identify the items and executing the
      reverse. This requires that only reversible transformations be provided
      here. The reversible transformations are: - `CryptoDeterministicConfig`
      - `CryptoReplaceFfxFpeConfig`
    reidentifyTemplateName: Template to use. References an instance of
      `DeidentifyTemplate`. Any configuration directly specified in
      `reidentify_config` or `inspect_config` will override those set in the
      template. The `DeidentifyTemplate` used must include only reversible
      transformations. Singular fields that are set in this request will
      replace their corresponding fields in the template. Repeated fields are
      appended. Singular sub-messages and groups are recursively merged.
  """
    inspectConfig = _messages.MessageField('GooglePrivacyDlpV2InspectConfig', 1)
    inspectTemplateName = _messages.StringField(2)
    item = _messages.MessageField('GooglePrivacyDlpV2ContentItem', 3)
    locationId = _messages.StringField(4)
    reidentifyConfig = _messages.MessageField('GooglePrivacyDlpV2DeidentifyConfig', 5)
    reidentifyTemplateName = _messages.StringField(6)