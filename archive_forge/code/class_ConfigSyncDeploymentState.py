from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigSyncDeploymentState(_messages.Message):
    """The state of ConfigSync's deployment on a cluster

  Enums:
    AdmissionWebhookValueValuesEnum: Deployment state of admission-webhook
    GitSyncValueValuesEnum: Deployment state of the git-sync pod
    ImporterValueValuesEnum: Deployment state of the importer pod
    MonitorValueValuesEnum: Deployment state of the monitor pod
    ReconcilerManagerValueValuesEnum: Deployment state of reconciler-manager
      pod
    RootReconcilerValueValuesEnum: Deployment state of root-reconciler
    SyncerValueValuesEnum: Deployment state of the syncer pod

  Fields:
    admissionWebhook: Deployment state of admission-webhook
    gitSync: Deployment state of the git-sync pod
    importer: Deployment state of the importer pod
    monitor: Deployment state of the monitor pod
    reconcilerManager: Deployment state of reconciler-manager pod
    rootReconciler: Deployment state of root-reconciler
    syncer: Deployment state of the syncer pod
  """

    class AdmissionWebhookValueValuesEnum(_messages.Enum):
        """Deployment state of admission-webhook

    Values:
      DEPLOYMENT_STATE_UNSPECIFIED: Deployment's state cannot be determined
      NOT_INSTALLED: Deployment is not installed
      INSTALLED: Deployment is installed
      ERROR: Deployment was attempted to be installed, but has errors
      PENDING: Deployment is installing or terminating
    """
        DEPLOYMENT_STATE_UNSPECIFIED = 0
        NOT_INSTALLED = 1
        INSTALLED = 2
        ERROR = 3
        PENDING = 4

    class GitSyncValueValuesEnum(_messages.Enum):
        """Deployment state of the git-sync pod

    Values:
      DEPLOYMENT_STATE_UNSPECIFIED: Deployment's state cannot be determined
      NOT_INSTALLED: Deployment is not installed
      INSTALLED: Deployment is installed
      ERROR: Deployment was attempted to be installed, but has errors
      PENDING: Deployment is installing or terminating
    """
        DEPLOYMENT_STATE_UNSPECIFIED = 0
        NOT_INSTALLED = 1
        INSTALLED = 2
        ERROR = 3
        PENDING = 4

    class ImporterValueValuesEnum(_messages.Enum):
        """Deployment state of the importer pod

    Values:
      DEPLOYMENT_STATE_UNSPECIFIED: Deployment's state cannot be determined
      NOT_INSTALLED: Deployment is not installed
      INSTALLED: Deployment is installed
      ERROR: Deployment was attempted to be installed, but has errors
      PENDING: Deployment is installing or terminating
    """
        DEPLOYMENT_STATE_UNSPECIFIED = 0
        NOT_INSTALLED = 1
        INSTALLED = 2
        ERROR = 3
        PENDING = 4

    class MonitorValueValuesEnum(_messages.Enum):
        """Deployment state of the monitor pod

    Values:
      DEPLOYMENT_STATE_UNSPECIFIED: Deployment's state cannot be determined
      NOT_INSTALLED: Deployment is not installed
      INSTALLED: Deployment is installed
      ERROR: Deployment was attempted to be installed, but has errors
      PENDING: Deployment is installing or terminating
    """
        DEPLOYMENT_STATE_UNSPECIFIED = 0
        NOT_INSTALLED = 1
        INSTALLED = 2
        ERROR = 3
        PENDING = 4

    class ReconcilerManagerValueValuesEnum(_messages.Enum):
        """Deployment state of reconciler-manager pod

    Values:
      DEPLOYMENT_STATE_UNSPECIFIED: Deployment's state cannot be determined
      NOT_INSTALLED: Deployment is not installed
      INSTALLED: Deployment is installed
      ERROR: Deployment was attempted to be installed, but has errors
      PENDING: Deployment is installing or terminating
    """
        DEPLOYMENT_STATE_UNSPECIFIED = 0
        NOT_INSTALLED = 1
        INSTALLED = 2
        ERROR = 3
        PENDING = 4

    class RootReconcilerValueValuesEnum(_messages.Enum):
        """Deployment state of root-reconciler

    Values:
      DEPLOYMENT_STATE_UNSPECIFIED: Deployment's state cannot be determined
      NOT_INSTALLED: Deployment is not installed
      INSTALLED: Deployment is installed
      ERROR: Deployment was attempted to be installed, but has errors
      PENDING: Deployment is installing or terminating
    """
        DEPLOYMENT_STATE_UNSPECIFIED = 0
        NOT_INSTALLED = 1
        INSTALLED = 2
        ERROR = 3
        PENDING = 4

    class SyncerValueValuesEnum(_messages.Enum):
        """Deployment state of the syncer pod

    Values:
      DEPLOYMENT_STATE_UNSPECIFIED: Deployment's state cannot be determined
      NOT_INSTALLED: Deployment is not installed
      INSTALLED: Deployment is installed
      ERROR: Deployment was attempted to be installed, but has errors
      PENDING: Deployment is installing or terminating
    """
        DEPLOYMENT_STATE_UNSPECIFIED = 0
        NOT_INSTALLED = 1
        INSTALLED = 2
        ERROR = 3
        PENDING = 4
    admissionWebhook = _messages.EnumField('AdmissionWebhookValueValuesEnum', 1)
    gitSync = _messages.EnumField('GitSyncValueValuesEnum', 2)
    importer = _messages.EnumField('ImporterValueValuesEnum', 3)
    monitor = _messages.EnumField('MonitorValueValuesEnum', 4)
    reconcilerManager = _messages.EnumField('ReconcilerManagerValueValuesEnum', 5)
    rootReconciler = _messages.EnumField('RootReconcilerValueValuesEnum', 6)
    syncer = _messages.EnumField('SyncerValueValuesEnum', 7)