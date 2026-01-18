from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeploymentGroupTypeValueValuesEnum(_messages.Enum):
    """Type of the deployment group, which will be either Standard or
    Extensible.

    Values:
      DEPLOYMENT_GROUP_TYPE_UNSPECIFIED: Unspecified type
      STANDARD: Standard type
      EXTENSIBLE: Extensible Type
    """
    DEPLOYMENT_GROUP_TYPE_UNSPECIFIED = 0
    STANDARD = 1
    EXTENSIBLE = 2