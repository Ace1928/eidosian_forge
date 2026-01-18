from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV1MuteConfig(_messages.Message):
    """A mute config is a Cloud SCC resource that contains the configuration to
  mute create/update events of findings.

  Fields:
    createTime: Output only. The time at which the mute config was created.
      This field is set by the server and will be ignored if provided on
      config creation.
    description: A description of the mute config.
    displayName: The human readable name to be displayed for the mute config.
    filter: Required. An expression that defines the filter to apply across
      create/update events of findings. While creating a filter string, be
      mindful of the scope in which the mute configuration is being created.
      E.g., If a filter contains project = X but is created under the project
      = Y scope, it might not match any findings. The following field and
      operator combinations are supported: * severity: `=`, `:` * category:
      `=`, `:` * resource.name: `=`, `:` * resource.project_name: `=`, `:` *
      resource.project_display_name: `=`, `:` *
      resource.folders.resource_folder: `=`, `:` * resource.parent_name: `=`,
      `:` * resource.parent_display_name: `=`, `:` * resource.type: `=`, `:` *
      finding_class: `=`, `:` * indicator.ip_addresses: `=`, `:` *
      indicator.domains: `=`, `:`
    mostRecentEditor: Output only. Email address of the user who last edited
      the mute config. This field is set by the server and will be ignored if
      provided on config creation or update.
    name: This field will be ignored if provided on config creation. Format
      "organizations/{organization}/muteConfigs/{mute_config}"
      "folders/{folder}/muteConfigs/{mute_config}"
      "projects/{project}/muteConfigs/{mute_config}" "organizations/{organizat
      ion}/locations/global/muteConfigs/{mute_config}"
      "folders/{folder}/locations/global/muteConfigs/{mute_config}"
      "projects/{project}/locations/global/muteConfigs/{mute_config}"
    updateTime: Output only. The most recent time at which the mute config was
      updated. This field is set by the server and will be ignored if provided
      on config creation or update.
  """
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)
    filter = _messages.StringField(4)
    mostRecentEditor = _messages.StringField(5)
    name = _messages.StringField(6)
    updateTime = _messages.StringField(7)