from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3Environment(_messages.Message):
    """Represents an environment for an agent. You can create multiple versions
  of your agent and publish them to separate environments. When you edit an
  agent, you are editing the draft agent. At any point, you can save the draft
  agent as an agent version, which is an immutable snapshot of your agent.
  When you save the draft agent, it is published to the default environment.
  When you create agent versions, you can publish them to custom environments.
  You can create a variety of custom environments for testing, development,
  production, etc.

  Fields:
    description: The human-readable description of the environment. The
      maximum length is 500 characters. If exceeded, the request is rejected.
    displayName: Required. The human-readable name of the environment (unique
      in an agent). Limit of 64 characters.
    name: The name of the environment. Format:
      `projects//locations//agents//environments/`.
    testCasesConfig: The test cases config for continuous tests of this
      environment.
    updateTime: Output only. Update time of this environment.
    versionConfigs: A list of configurations for flow versions. You should
      include version configs for all flows that are reachable from `Start
      Flow` in the agent. Otherwise, an error will be returned.
    webhookConfig: The webhook configuration for this environment.
  """
    description = _messages.StringField(1)
    displayName = _messages.StringField(2)
    name = _messages.StringField(3)
    testCasesConfig = _messages.MessageField('GoogleCloudDialogflowCxV3EnvironmentTestCasesConfig', 4)
    updateTime = _messages.StringField(5)
    versionConfigs = _messages.MessageField('GoogleCloudDialogflowCxV3EnvironmentVersionConfig', 6, repeated=True)
    webhookConfig = _messages.MessageField('GoogleCloudDialogflowCxV3EnvironmentWebhookConfig', 7)