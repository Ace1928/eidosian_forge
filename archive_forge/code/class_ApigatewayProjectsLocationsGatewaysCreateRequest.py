from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigatewayProjectsLocationsGatewaysCreateRequest(_messages.Message):
    """A ApigatewayProjectsLocationsGatewaysCreateRequest object.

  Fields:
    apigatewayGateway: A ApigatewayGateway resource to be passed as the
      request body.
    gatewayId: Required. Identifier to assign to the Gateway. Must be unique
      within scope of the parent resource.
    parent: Required. Parent resource of the Gateway, of the form:
      `projects/*/locations/*`
  """
    apigatewayGateway = _messages.MessageField('ApigatewayGateway', 1)
    gatewayId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)