from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterOrganizationsMuteConfigsDeleteRequest(_messages.Message):
    """A SecuritycenterOrganizationsMuteConfigsDeleteRequest object.

  Fields:
    name: Required. Name of the mute config to delete. The following list
      shows some examples of the format: +
      `organizations/{organization}/muteConfigs/{config_id}` + `organizations/
      {organization}/locations/{location}/muteConfigs/{config_id}` +
      `folders/{folder}/muteConfigs/{config_id}` +
      `folders/{folder}/locations/{location}/muteConfigs/{config_id}` +
      `projects/{project}/muteConfigs/{config_id}` +
      `projects/{project}/locations/{location}/muteConfigs/{config_id}`
  """
    name = _messages.StringField(1, required=True)