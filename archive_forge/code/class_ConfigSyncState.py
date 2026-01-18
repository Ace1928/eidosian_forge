from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigSyncState(_messages.Message):
    """State information for ConfigSync

  Enums:
    ReposyncCrdValueValuesEnum: The state of the Reposync CRD
    RootsyncCrdValueValuesEnum: The state of the RootSync CRD
    StateValueValuesEnum: The state of CS This field summarizes the other
      fields in this message.

  Fields:
    deploymentState: Information about the deployment of ConfigSync, including
      the version of the various Pods deployed
    errors: Errors pertaining to the installation of Config Sync.
    reposyncCrd: The state of the Reposync CRD
    rootsyncCrd: The state of the RootSync CRD
    state: The state of CS This field summarizes the other fields in this
      message.
    syncState: The state of ConfigSync's process to sync configs to a cluster
    version: The version of ConfigSync deployed
  """

    class ReposyncCrdValueValuesEnum(_messages.Enum):
        """The state of the Reposync CRD

    Values:
      CRD_STATE_UNSPECIFIED: CRD's state cannot be determined
      NOT_INSTALLED: CRD is not installed
      INSTALLED: CRD is installed
      TERMINATING: CRD is terminating (i.e., it has been deleted and is
        cleaning up)
      INSTALLING: CRD is installing
    """
        CRD_STATE_UNSPECIFIED = 0
        NOT_INSTALLED = 1
        INSTALLED = 2
        TERMINATING = 3
        INSTALLING = 4

    class RootsyncCrdValueValuesEnum(_messages.Enum):
        """The state of the RootSync CRD

    Values:
      CRD_STATE_UNSPECIFIED: CRD's state cannot be determined
      NOT_INSTALLED: CRD is not installed
      INSTALLED: CRD is installed
      TERMINATING: CRD is terminating (i.e., it has been deleted and is
        cleaning up)
      INSTALLING: CRD is installing
    """
        CRD_STATE_UNSPECIFIED = 0
        NOT_INSTALLED = 1
        INSTALLED = 2
        TERMINATING = 3
        INSTALLING = 4

    class StateValueValuesEnum(_messages.Enum):
        """The state of CS This field summarizes the other fields in this
    message.

    Values:
      STATE_UNSPECIFIED: CS's state cannot be determined.
      CONFIG_SYNC_NOT_INSTALLED: CS is not installed.
      CONFIG_SYNC_INSTALLED: The expected CS version is installed
        successfully.
      CONFIG_SYNC_ERROR: CS encounters errors.
      CONFIG_SYNC_PENDING: CS is installing or terminating.
    """
        STATE_UNSPECIFIED = 0
        CONFIG_SYNC_NOT_INSTALLED = 1
        CONFIG_SYNC_INSTALLED = 2
        CONFIG_SYNC_ERROR = 3
        CONFIG_SYNC_PENDING = 4
    deploymentState = _messages.MessageField('ConfigSyncDeploymentState', 1)
    errors = _messages.MessageField('ConfigSyncError', 2, repeated=True)
    reposyncCrd = _messages.EnumField('ReposyncCrdValueValuesEnum', 3)
    rootsyncCrd = _messages.EnumField('RootsyncCrdValueValuesEnum', 4)
    state = _messages.EnumField('StateValueValuesEnum', 5)
    syncState = _messages.MessageField('SyncState', 6)
    version = _messages.MessageField('ConfigSyncVersion', 7)