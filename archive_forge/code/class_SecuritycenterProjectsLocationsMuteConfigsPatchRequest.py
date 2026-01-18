from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterProjectsLocationsMuteConfigsPatchRequest(_messages.Message):
    """A SecuritycenterProjectsLocationsMuteConfigsPatchRequest object.

  Fields:
    googleCloudSecuritycenterV2MuteConfig: A
      GoogleCloudSecuritycenterV2MuteConfig resource to be passed as the
      request body.
    name: This field will be ignored if provided on config creation. The
      following list shows some examples of the format: +
      `organizations/{organization}/muteConfigs/{mute_config}` + `organization
      s/{organization}locations/{location}//muteConfigs/{mute_config}` +
      `folders/{folder}/muteConfigs/{mute_config}` +
      `folders/{folder}/locations/{location}/muteConfigs/{mute_config}` +
      `projects/{project}/muteConfigs/{mute_config}` +
      `projects/{project}/locations/{location}/muteConfigs/{mute_config}`
    updateMask: The list of fields to be updated. If empty all mutable fields
      will be updated.
  """
    googleCloudSecuritycenterV2MuteConfig = _messages.MessageField('GoogleCloudSecuritycenterV2MuteConfig', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)