from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SharedDeploymentPolicyValueValuesEnum(_messages.Enum):
    """Policy configuration about how user applications are deployed.

    Values:
      SHARED_DEPLOYMENT_POLICY_UNSPECIFIED: Unspecified.
      ALLOWED: User applications can be deployed both on control plane and
        worker nodes.
      DISALLOWED: User applications can not be deployed on control plane nodes
        and can only be deployed on worker nodes.
    """
    SHARED_DEPLOYMENT_POLICY_UNSPECIFIED = 0
    ALLOWED = 1
    DISALLOWED = 2