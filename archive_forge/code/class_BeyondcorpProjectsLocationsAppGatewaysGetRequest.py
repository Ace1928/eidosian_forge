from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BeyondcorpProjectsLocationsAppGatewaysGetRequest(_messages.Message):
    """A BeyondcorpProjectsLocationsAppGatewaysGetRequest object.

  Fields:
    name: Required. BeyondCorp AppGateway name using the form: `projects/{proj
      ect_id}/locations/{location_id}/appGateways/{app_gateway_id}`
  """
    name = _messages.StringField(1, required=True)