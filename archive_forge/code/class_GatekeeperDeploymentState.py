from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GatekeeperDeploymentState(_messages.Message):
    """State of Policy Controller installation.

  Enums:
    GatekeeperAuditValueValuesEnum: Status of gatekeeper-audit deployment.
    GatekeeperControllerManagerStateValueValuesEnum: Status of gatekeeper-
      controller-manager pod.
    GatekeeperMutationValueValuesEnum: Status of the pod serving the mutation
      webhook.

  Fields:
    gatekeeperAudit: Status of gatekeeper-audit deployment.
    gatekeeperControllerManagerState: Status of gatekeeper-controller-manager
      pod.
    gatekeeperMutation: Status of the pod serving the mutation webhook.
  """

    class GatekeeperAuditValueValuesEnum(_messages.Enum):
        """Status of gatekeeper-audit deployment.

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

    class GatekeeperControllerManagerStateValueValuesEnum(_messages.Enum):
        """Status of gatekeeper-controller-manager pod.

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

    class GatekeeperMutationValueValuesEnum(_messages.Enum):
        """Status of the pod serving the mutation webhook.

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
    gatekeeperAudit = _messages.EnumField('GatekeeperAuditValueValuesEnum', 1)
    gatekeeperControllerManagerState = _messages.EnumField('GatekeeperControllerManagerStateValueValuesEnum', 2)
    gatekeeperMutation = _messages.EnumField('GatekeeperMutationValueValuesEnum', 3)