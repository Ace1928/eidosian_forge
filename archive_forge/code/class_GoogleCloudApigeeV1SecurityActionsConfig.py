from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1SecurityActionsConfig(_messages.Message):
    """SecurityActionsConfig reflects the current state of the SecurityActions
  feature. This is a singleton resource: https://google.aip.dev/156

  Fields:
    enabled: The flag that controls whether this feature is enabled. This is
      `unset` by default. When this flag is `false`, even if individual rules
      are enabled, no SecurityActions will be enforced.
    name: This is a singleton resource, the name will always be set by
      SecurityActions and any user input will be ignored. The name is always:
      `organizations/{org}/environments/{env}/security_actions_config`
    updateTime: Output only. The update time for configuration.
  """
    enabled = _messages.BooleanField(1)
    name = _messages.StringField(2)
    updateTime = _messages.StringField(3)