from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisGoogleDevtoolsCloudbuildV1Source(_messages.Message):
    """Location of the source in a supported storage service.

  Fields:
    connectedRepository: Optional. If provided, get the source from this 2nd-
      gen Google Cloud Build repository resource.
    gitSource: If provided, get the source from this Git repository.
    repoSource: If provided, get the source from this location in a Cloud
      Source Repository.
    storageSource: If provided, get the source from this location in Cloud
      Storage.
    storageSourceManifest: If provided, get the source from this manifest in
      Cloud Storage. This feature is in Preview; see description
      [here](https://github.com/GoogleCloudPlatform/cloud-
      builders/tree/master/gcs-fetcher).
  """
    connectedRepository = _messages.MessageField('ContaineranalysisGoogleDevtoolsCloudbuildV1ConnectedRepository', 1)
    gitSource = _messages.MessageField('ContaineranalysisGoogleDevtoolsCloudbuildV1GitSource', 2)
    repoSource = _messages.MessageField('ContaineranalysisGoogleDevtoolsCloudbuildV1RepoSource', 3)
    storageSource = _messages.MessageField('ContaineranalysisGoogleDevtoolsCloudbuildV1StorageSource', 4)
    storageSourceManifest = _messages.MessageField('ContaineranalysisGoogleDevtoolsCloudbuildV1StorageSourceManifest', 5)