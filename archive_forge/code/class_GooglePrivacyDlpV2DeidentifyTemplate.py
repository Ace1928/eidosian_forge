from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DeidentifyTemplate(_messages.Message):
    """DeidentifyTemplates contains instructions on how to de-identify content.
  See https://cloud.google.com/sensitive-data-protection/docs/concepts-
  templates to learn more.

  Fields:
    createTime: Output only. The creation timestamp of an inspectTemplate.
    deidentifyConfig: The core content of the template.
    description: Short description (max 256 chars).
    displayName: Display name (max 256 chars).
    name: Output only. The template name. The template will have one of the
      following formats: `projects/PROJECT_ID/deidentifyTemplates/TEMPLATE_ID`
      OR `organizations/ORGANIZATION_ID/deidentifyTemplates/TEMPLATE_ID`
    updateTime: Output only. The last update timestamp of an inspectTemplate.
  """
    createTime = _messages.StringField(1)
    deidentifyConfig = _messages.MessageField('GooglePrivacyDlpV2DeidentifyConfig', 2)
    description = _messages.StringField(3)
    displayName = _messages.StringField(4)
    name = _messages.StringField(5)
    updateTime = _messages.StringField(6)