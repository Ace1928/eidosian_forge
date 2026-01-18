from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeExternalVpnGatewaysGetRequest(_messages.Message):
    """A ComputeExternalVpnGatewaysGetRequest object.

  Fields:
    externalVpnGateway: Name of the externalVpnGateway to return.
    project: Project ID for this request.
  """
    externalVpnGateway = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)