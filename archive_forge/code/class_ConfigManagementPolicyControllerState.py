from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigManagementPolicyControllerState(_messages.Message):
    """State for PolicyControllerState.

  Fields:
    deploymentState: The state about the policy controller installation.
    migration: Record state of ACM -> PoCo Hub migration for this feature.
    version: The version of Gatekeeper Policy Controller deployed.
  """
    deploymentState = _messages.MessageField('ConfigManagementGatekeeperDeploymentState', 1)
    migration = _messages.MessageField('ConfigManagementPolicyControllerMigration', 2)
    version = _messages.MessageField('ConfigManagementPolicyControllerVersion', 3)