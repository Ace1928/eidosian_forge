from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RemoteRepositoryConfig(_messages.Message):
    """Remote repository configuration.

  Enums:
    RemoteTypeValueValuesEnum:

  Fields:
    aptRepository: Specific settings for an Apt remote repository.
    deleteNotFoundCacheFiles: Optional. If files are removed from the remote
      host, should they also be removed from the Artifact Registry repository
      when requested? Only supported for docker, maven, and python
    description: The description of the remote source.
    disableUpstreamValidation: Input only. A create/update remote repo option
      to avoid making a HEAD/GET request to validate a remote repo and any
      supplied upstream credentials.
    dockerRepository: Specific settings for a Docker remote repository.
    goRepository: Specific settings for a Go remote repository.
    mavenRepository: Specific settings for a Maven remote repository.
    npmRepository: Specific settings for an Npm remote repository.
    pythonRepository: Specific settings for a Python remote repository.
    remoteType: A RemoteTypeValueValuesEnum attribute.
    upstreamCredentials: Optional. The credentials used to access the remote
      repository.
    yumRepository: Specific settings for a Yum remote repository.
  """

    class RemoteTypeValueValuesEnum(_messages.Enum):
        """RemoteTypeValueValuesEnum enum type.

    Values:
      REMOTE_TYPE_UNSPECIFIED: <no description>
      MIRROR: <no description>
      CACHE_LAYER: <no description>
    """
        REMOTE_TYPE_UNSPECIFIED = 0
        MIRROR = 1
        CACHE_LAYER = 2
    aptRepository = _messages.MessageField('AptRepository', 1)
    deleteNotFoundCacheFiles = _messages.BooleanField(2)
    description = _messages.StringField(3)
    disableUpstreamValidation = _messages.BooleanField(4)
    dockerRepository = _messages.MessageField('DockerRepository', 5)
    goRepository = _messages.MessageField('GoRepository', 6)
    mavenRepository = _messages.MessageField('MavenRepository', 7)
    npmRepository = _messages.MessageField('NpmRepository', 8)
    pythonRepository = _messages.MessageField('PythonRepository', 9)
    remoteType = _messages.EnumField('RemoteTypeValueValuesEnum', 10)
    upstreamCredentials = _messages.MessageField('UpstreamCredentials', 11)
    yumRepository = _messages.MessageField('YumRepository', 12)