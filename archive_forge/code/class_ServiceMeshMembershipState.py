from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceMeshMembershipState(_messages.Message):
    """**Service Mesh**: State for a single Membership, as analyzed by the
  Service Mesh Hub Controller.

  Fields:
    conditions: Output only. List of condition reporting membership statues
    controlPlaneManagement: Output only. Status of control plane management
    dataPlaneManagement: Output only. Status of data plane management.
  """
    conditions = _messages.MessageField('ServiceMeshCondition', 1, repeated=True)
    controlPlaneManagement = _messages.MessageField('ServiceMeshControlPlaneManagement', 2)
    dataPlaneManagement = _messages.MessageField('ServiceMeshDataPlaneManagement', 3)