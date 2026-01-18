from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsBitbucketServerConfigsConnectedRepositoriesBatchCreateRequest(_messages.Message):
    """A CloudbuildProjectsLocationsBitbucketServerConfigsConnectedRepositories
  BatchCreateRequest object.

  Fields:
    batchCreateBitbucketServerConnectedRepositoriesRequest: A
      BatchCreateBitbucketServerConnectedRepositoriesRequest resource to be
      passed as the request body.
    parent: The name of the `BitbucketServerConfig` that added connected
      repository. Format: `projects/{project}/locations/{location}/bitbucketSe
      rverConfigs/{config}`
  """
    batchCreateBitbucketServerConnectedRepositoriesRequest = _messages.MessageField('BatchCreateBitbucketServerConnectedRepositoriesRequest', 1)
    parent = _messages.StringField(2, required=True)