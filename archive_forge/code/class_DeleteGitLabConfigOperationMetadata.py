from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeleteGitLabConfigOperationMetadata(_messages.Message):
    """Metadata for `DeleteGitLabConfig` operation.

  Fields:
    completeTime: Time the operation was completed.
    createTime: Time the operation was created.
    gitlabConfig: The resource name of the GitLabConfig to be created. Format:
      `projects/{project}/locations/{location}/gitlabConfigs/{id}`.
  """
    completeTime = _messages.StringField(1)
    createTime = _messages.StringField(2)
    gitlabConfig = _messages.StringField(3)