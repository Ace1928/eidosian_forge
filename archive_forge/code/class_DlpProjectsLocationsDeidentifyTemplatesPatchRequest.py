from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DlpProjectsLocationsDeidentifyTemplatesPatchRequest(_messages.Message):
    """A DlpProjectsLocationsDeidentifyTemplatesPatchRequest object.

  Fields:
    googlePrivacyDlpV2UpdateDeidentifyTemplateRequest: A
      GooglePrivacyDlpV2UpdateDeidentifyTemplateRequest resource to be passed
      as the request body.
    name: Required. Resource name of organization and deidentify template to
      be updated, for example
      `organizations/433245324/deidentifyTemplates/432452342` or
      projects/project-id/deidentifyTemplates/432452342.
  """
    googlePrivacyDlpV2UpdateDeidentifyTemplateRequest = _messages.MessageField('GooglePrivacyDlpV2UpdateDeidentifyTemplateRequest', 1)
    name = _messages.StringField(2, required=True)