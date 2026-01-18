from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2MuteConfig(_messages.Message):
    """A mute config is a Cloud SCC resource that contains the configuration to
  mute create/update events of findings.

  Enums:
    TypeValueValuesEnum: Required. The type of the mute config, which
      determines what type of mute state the config affects. Immutable after
      creation.

  Fields:
    createTime: Output only. The time at which the mute config was created.
      This field is set by the server and will be ignored if provided on
      config creation.
    description: A description of the mute config.
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
    name: This field will be ignored if provided on config creation. The
      following list shows some examples of the format: +
      `organizations/{organization}/muteConfigs/{mute_config}` + `organization
      s/{organization}locations/{location}//muteConfigs/{mute_config}` +
      `folders/{folder}/muteConfigs/{mute_config}` +
      `folders/{folder}/locations/{location}/muteConfigs/{mute_config}` +
      `projects/{project}/muteConfigs/{mute_config}` +
      `projects/{project}/locations/{location}/muteConfigs/{mute_config}`
    type: Required. The type of the mute config, which determines what type of
      mute state the config affects. Immutable after creation.
    updateTime: Output only. The most recent time at which the mute config was
      updated. This field is set by the server and will be ignored if provided
      on config creation or update.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Required. The type of the mute config, which determines what type of
    mute state the config affects. Immutable after creation.

    Values:
      MUTE_CONFIG_TYPE_UNSPECIFIED: Unused.
      STATIC: A static mute config, which sets the static mute state of future
        matching findings to muted. Once the static mute state has been set,
        finding or config modifications will not affect the state.
    """
        MUTE_CONFIG_TYPE_UNSPECIFIED = 0
        STATIC = 1
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    filter = _messages.StringField(3)
    mostRecentEditor = _messages.StringField(4)
    name = _messages.StringField(5)
    type = _messages.EnumField('TypeValueValuesEnum', 6)
    updateTime = _messages.StringField(7)