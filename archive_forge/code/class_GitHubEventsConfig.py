from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GitHubEventsConfig(_messages.Message):
    """GitHubEventsConfig describes the configuration of a trigger that creates
  a build whenever a GitHub event is received.

  Fields:
    enterpriseConfig: Output only. The GitHubEnterpriseConfig enterprise
      config specified in the enterprise_config_resource_name field.
    enterpriseConfigResourceName: Optional. The resource name of the github
      enterprise config that should be applied to this installation. For
      example: "projects/{$project_id}/locations/{$location_id}/githubEnterpri
      seConfigs/{$config_id}"
    installationId: The installationID that emits the GitHub event.
    name: Name of the repository. For example: The name for
      https://github.com/googlecloudplatform/cloud-builders is "cloud-
      builders".
    owner: Owner of the repository. For example: The owner for
      https://github.com/googlecloudplatform/cloud-builders is
      "googlecloudplatform".
    pullRequest: filter to match changes in pull requests.
    push: filter to match changes in refs like branches, tags.
  """
    enterpriseConfig = _messages.MessageField('GitHubEnterpriseConfig', 1)
    enterpriseConfigResourceName = _messages.StringField(2)
    installationId = _messages.IntegerField(3)
    name = _messages.StringField(4)
    owner = _messages.StringField(5)
    pullRequest = _messages.MessageField('PullRequestFilter', 6)
    push = _messages.MessageField('PushFilter', 7)