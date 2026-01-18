from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PostureDetails(_messages.Message):
    """Details of a posture deployment.

  Fields:
    policySet: ID of the above posture's policy set to which this policy
      belongs.
    posture: Posture name in the format of organizations/{organization_id}/loc
      ations/{location_id}/postures/{postureID}.
    postureDeployment: Posture deployment name in one of the following formats
      organizations/{organization_id}/locations/{location_id}/postureDeploymen
      ts/{postureDeploymentID}
    postureDeploymentTargetResource: Target resource where the Posture is
      deployed. Can be one of: projects/projectNumber, folders/folderNumber,
      organizations/organizationNumber.
    postureRevisionId: Posture revision ID.
  """
    policySet = _messages.StringField(1)
    posture = _messages.StringField(2)
    postureDeployment = _messages.StringField(3)
    postureDeploymentTargetResource = _messages.StringField(4)
    postureRevisionId = _messages.StringField(5)