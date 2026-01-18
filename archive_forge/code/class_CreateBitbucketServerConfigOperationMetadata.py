from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateBitbucketServerConfigOperationMetadata(_messages.Message):
    """Metadata for `CreateBitbucketServerConfig` operation.

  Fields:
    bitbucketServerConfig: The resource name of the BitbucketServerConfig to
      be created. Format:
      `projects/{project}/locations/{location}/bitbucketServerConfigs/{id}`.
    completeTime: Time the operation was completed.
    createTime: Time the operation was created.
  """
    bitbucketServerConfig = _messages.StringField(1)
    completeTime = _messages.StringField(2)
    createTime = _messages.StringField(3)