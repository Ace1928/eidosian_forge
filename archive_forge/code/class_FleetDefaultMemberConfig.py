from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FleetDefaultMemberConfig(_messages.Message):
    """FleetDefaultMemberConfig contains default configuration information for
  memberships of a fleet.

  Fields:
    identityService: Spec for IdentityService.
    serviceMesh: Spec for ServiceMesh.
  """
    identityService = _messages.MessageField('MemberConfig', 1)
    serviceMesh = _messages.MessageField('ServiceMeshMembershipSpec', 2)