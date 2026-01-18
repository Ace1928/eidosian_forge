from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NpmRepository(_messages.Message):
    """Configuration for a Npm remote repository.

  Enums:
    PublicRepositoryValueValuesEnum: One of the publicly available Npm
      repositories supported by Artifact Registry.

  Fields:
    artifactRegistryRepository: An Artifact Registry Repository.
    customRepository: Customer-specified remote repository.
    publicRepository: One of the publicly available Npm repositories supported
      by Artifact Registry.
  """

    class PublicRepositoryValueValuesEnum(_messages.Enum):
        """One of the publicly available Npm repositories supported by Artifact
    Registry.

    Values:
      PUBLIC_REPOSITORY_UNSPECIFIED: Unspecified repository.
      NPMJS: npmjs.
    """
        PUBLIC_REPOSITORY_UNSPECIFIED = 0
        NPMJS = 1
    artifactRegistryRepository = _messages.MessageField('GoogleDevtoolsArtifactregistryV1RemoteRepositoryConfigNpmRepositoryArtifactRegistryRepository', 1)
    customRepository = _messages.MessageField('GoogleDevtoolsArtifactregistryV1RemoteRepositoryConfigNpmRepositoryCustomRepository', 2)
    publicRepository = _messages.EnumField('PublicRepositoryValueValuesEnum', 3)