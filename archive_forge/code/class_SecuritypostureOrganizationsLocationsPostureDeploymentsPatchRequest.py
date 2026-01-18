from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritypostureOrganizationsLocationsPostureDeploymentsPatchRequest(_messages.Message):
    """A SecuritypostureOrganizationsLocationsPostureDeploymentsPatchRequest
  object.

  Fields:
    name: Required. The name of this PostureDeployment resource, in the format
      of organizations/{organization}/locations/{location_id}/postureDeploymen
      ts/{postureDeployment}.
    postureDeployment: A PostureDeployment resource to be passed as the
      request body.
    updateMask: Required. Field mask is used to specify the fields to be
      overwritten in the PostureDeployment resource by the update. The fields
      specified in the update_mask are relative to the resource, not the full
      request. A field will be overwritten if it is in the mask. If the user
      does not provide a mask then all fields will be overwritten.
  """
    name = _messages.StringField(1, required=True)
    postureDeployment = _messages.MessageField('PostureDeployment', 2)
    updateMask = _messages.StringField(3)