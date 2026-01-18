from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigatewayProjectsLocationsApisConfigsPatchRequest(_messages.Message):
    """A ApigatewayProjectsLocationsApisConfigsPatchRequest object.

  Fields:
    apigatewayApiConfig: A ApigatewayApiConfig resource to be passed as the
      request body.
    name: Output only. Resource name of the API Config. Format:
      projects/{project}/locations/global/apis/{api}/configs/{api_config}
    updateMask: Field mask is used to specify the fields to be overwritten in
      the ApiConfig resource by the update. The fields specified in the
      update_mask are relative to the resource, not the full request. A field
      will be overwritten if it is in the mask. If the user does not provide a
      mask then all fields will be overwritten.
  """
    apigatewayApiConfig = _messages.MessageField('ApigatewayApiConfig', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)