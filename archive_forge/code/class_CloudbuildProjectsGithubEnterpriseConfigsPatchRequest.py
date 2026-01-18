from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsGithubEnterpriseConfigsPatchRequest(_messages.Message):
    """A CloudbuildProjectsGithubEnterpriseConfigsPatchRequest object.

  Fields:
    gitHubEnterpriseConfig: A GitHubEnterpriseConfig resource to be passed as
      the request body.
    name: Optional. The full resource name for the GitHubEnterpriseConfig For
      example: "projects/{$project_id}/locations/{$location_id}/githubEnterpri
      seConfigs/{$config_id}"
    updateMask: Update mask for the resource. If this is set, the server will
      only update the fields specified in the field mask. Otherwise, a full
      update of the mutable resource fields will be performed.
  """
    gitHubEnterpriseConfig = _messages.MessageField('GitHubEnterpriseConfig', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)