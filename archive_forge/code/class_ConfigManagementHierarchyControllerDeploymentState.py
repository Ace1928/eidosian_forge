from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigManagementHierarchyControllerDeploymentState(_messages.Message):
    """Deployment state for Hierarchy Controller

  Enums:
    ExtensionValueValuesEnum: The deployment state for Hierarchy Controller
      extension (e.g. v0.7.0-hc.1)
    HncValueValuesEnum: The deployment state for open source HNC (e.g.
      v0.7.0-hc.0)

  Fields:
    extension: The deployment state for Hierarchy Controller extension (e.g.
      v0.7.0-hc.1)
    hnc: The deployment state for open source HNC (e.g. v0.7.0-hc.0)
  """

    class ExtensionValueValuesEnum(_messages.Enum):
        """The deployment state for Hierarchy Controller extension (e.g.
    v0.7.0-hc.1)

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

    class HncValueValuesEnum(_messages.Enum):
        """The deployment state for open source HNC (e.g. v0.7.0-hc.0)

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
    extension = _messages.EnumField('ExtensionValueValuesEnum', 1)
    hnc = _messages.EnumField('HncValueValuesEnum', 2)