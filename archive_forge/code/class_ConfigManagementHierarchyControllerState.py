from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigManagementHierarchyControllerState(_messages.Message):
    """State for Hierarchy Controller

  Fields:
    state: The deployment state for Hierarchy Controller
    version: The version for Hierarchy Controller
  """
    state = _messages.MessageField('ConfigManagementHierarchyControllerDeploymentState', 1)
    version = _messages.MessageField('ConfigManagementHierarchyControllerVersion', 2)