from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsArtifactregistryV1RemoteRepositoryConfigYumRepositoryPublicRepository(_messages.Message):
    """Publicly available Yum repositories constructed from a common repository
  base and a custom repository path.

  Enums:
    RepositoryBaseValueValuesEnum: A common public repository base for Yum.

  Fields:
    repositoryBase: A common public repository base for Yum.
    repositoryPath: A custom field to define a path to a specific repository
      from the base.
  """

    class RepositoryBaseValueValuesEnum(_messages.Enum):
        """A common public repository base for Yum.

    Values:
      REPOSITORY_BASE_UNSPECIFIED: Unspecified repository base.
      CENTOS: CentOS.
      CENTOS_DEBUG: CentOS Debug.
      CENTOS_VAULT: CentOS Vault.
      CENTOS_STREAM: CentOS Stream.
      ROCKY: Rocky.
      EPEL: Fedora Extra Packages for Enterprise Linux (EPEL).
    """
        REPOSITORY_BASE_UNSPECIFIED = 0
        CENTOS = 1
        CENTOS_DEBUG = 2
        CENTOS_VAULT = 3
        CENTOS_STREAM = 4
        ROCKY = 5
        EPEL = 6
    repositoryBase = _messages.EnumField('RepositoryBaseValueValuesEnum', 1)
    repositoryPath = _messages.StringField(2)