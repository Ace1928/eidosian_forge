from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DataProfileConfigSnapshot(_messages.Message):
    """Snapshot of the configurations used to generate the profile.

  Fields:
    dataProfileJob: A copy of the configuration used to generate this profile.
      This is deprecated, and the DiscoveryConfig field is preferred moving
      forward. DataProfileJobConfig will still be written here for Discovery
      in BigQuery for backwards compatibility, but will not be updated with
      new fields, while DiscoveryConfig will.
    discoveryConfig: A copy of the configuration used to generate this
      profile.
    inspectConfig: A copy of the inspection config used to generate this
      profile. This is a copy of the inspect_template specified in
      `DataProfileJobConfig`.
    inspectTemplateModifiedTime: Timestamp when the template was modified
    inspectTemplateName: Name of the inspection template used to generate this
      profile
  """
    dataProfileJob = _messages.MessageField('GooglePrivacyDlpV2DataProfileJobConfig', 1)
    discoveryConfig = _messages.MessageField('GooglePrivacyDlpV2DiscoveryConfig', 2)
    inspectConfig = _messages.MessageField('GooglePrivacyDlpV2InspectConfig', 3)
    inspectTemplateModifiedTime = _messages.StringField(4)
    inspectTemplateName = _messages.StringField(5)