from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudiotProjectsLocationsRegistriesBindDeviceToGatewayRequest(_messages.Message):
    """A CloudiotProjectsLocationsRegistriesBindDeviceToGatewayRequest object.

  Fields:
    bindDeviceToGatewayRequest: A BindDeviceToGatewayRequest resource to be
      passed as the request body.
    parent: Required. The name of the registry. For example,
      `projects/example-project/locations/us-central1/registries/my-registry`.
  """
    bindDeviceToGatewayRequest = _messages.MessageField('BindDeviceToGatewayRequest', 1)
    parent = _messages.StringField(2, required=True)