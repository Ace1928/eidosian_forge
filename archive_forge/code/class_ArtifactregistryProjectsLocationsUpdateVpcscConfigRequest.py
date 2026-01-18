from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArtifactregistryProjectsLocationsUpdateVpcscConfigRequest(_messages.Message):
    """A ArtifactregistryProjectsLocationsUpdateVpcscConfigRequest object.

  Fields:
    name: The name of the project's VPC SC Config. Always of the form:
      projects/{projectID}/locations/{location}/vpcscConfig In update request:
      never set In response: always set
    updateMask: Field mask to support partial updates.
    vPCSCConfig: A VPCSCConfig resource to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    updateMask = _messages.StringField(2)
    vPCSCConfig = _messages.MessageField('VPCSCConfig', 3)