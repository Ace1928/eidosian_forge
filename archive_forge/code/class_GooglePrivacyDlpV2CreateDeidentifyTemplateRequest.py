from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2CreateDeidentifyTemplateRequest(_messages.Message):
    """Request message for CreateDeidentifyTemplate.

  Fields:
    deidentifyTemplate: Required. The DeidentifyTemplate to create.
    locationId: Deprecated. This field has no effect.
    templateId: The template id can contain uppercase and lowercase letters,
      numbers, and hyphens; that is, it must match the regular expression:
      `[a-zA-Z\\d-_]+`. The maximum length is 100 characters. Can be empty to
      allow the system to generate one.
  """
    deidentifyTemplate = _messages.MessageField('GooglePrivacyDlpV2DeidentifyTemplate', 1)
    locationId = _messages.StringField(2)
    templateId = _messages.StringField(3)