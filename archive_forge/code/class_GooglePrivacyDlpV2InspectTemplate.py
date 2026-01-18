from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2InspectTemplate(_messages.Message):
    """The inspectTemplate contains a configuration (set of types of sensitive
  data to be detected) to be used anywhere you otherwise would normally
  specify InspectConfig. See https://cloud.google.com/sensitive-data-
  protection/docs/concepts-templates to learn more.

  Fields:
    createTime: Output only. The creation timestamp of an inspectTemplate.
    description: Short description (max 256 chars).
    displayName: Display name (max 256 chars).
    inspectConfig: The core content of the template. Configuration of the
      scanning process.
    name: Output only. The template name. The template will have one of the
      following formats: `projects/PROJECT_ID/inspectTemplates/TEMPLATE_ID` OR
      `organizations/ORGANIZATION_ID/inspectTemplates/TEMPLATE_ID`;
    updateTime: Output only. The last update timestamp of an inspectTemplate.
  """
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)
    inspectConfig = _messages.MessageField('GooglePrivacyDlpV2InspectConfig', 4)
    name = _messages.StringField(5)
    updateTime = _messages.StringField(6)