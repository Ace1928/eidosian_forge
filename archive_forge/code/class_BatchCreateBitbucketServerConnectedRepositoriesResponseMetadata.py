from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchCreateBitbucketServerConnectedRepositoriesResponseMetadata(_messages.Message):
    """Metadata for `BatchCreateBitbucketServerConnectedRepositories`
  operation.

  Fields:
    completeTime: Time the operation was completed.
    config: The name of the `BitbucketServerConfig` that added connected
      repositories. Format: `projects/{project}/locations/{location}/bitbucket
      ServerConfigs/{config}`
    createTime: Time the operation was created.
  """
    completeTime = _messages.StringField(1)
    config = _messages.StringField(2)
    createTime = _messages.StringField(3)