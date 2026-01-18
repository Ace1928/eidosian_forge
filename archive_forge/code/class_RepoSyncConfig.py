from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class RepoSyncConfig(_messages.Message):
    """RepoSync configuration information.

  Enums:
    StatusValueValuesEnum: The status of RepoSync.

  Fields:
    externalRepoUrl: If this repo is enabled for RepoSync, this will be the
      URL of the external repo that this repo should sync with.
    status: The status of RepoSync.
  """

    class StatusValueValuesEnum(_messages.Enum):
        """The status of RepoSync.

    Values:
      REPO_SYNC_STATUS_UNSPECIFIED: No RepoSync status was specified.
      OK: RepoSync is working.
      FAILED_AUTH: RepoSync failed because of authorization/authentication.
      FAILED_OTHER: RepoSync failed for a reason other than auth.
      FAILED_NOT_FOUND: RepoSync failed because the repository was not found.
    """
        REPO_SYNC_STATUS_UNSPECIFIED = 0
        OK = 1
        FAILED_AUTH = 2
        FAILED_OTHER = 3
        FAILED_NOT_FOUND = 4
    externalRepoUrl = _messages.StringField(1)
    status = _messages.EnumField('StatusValueValuesEnum', 2)