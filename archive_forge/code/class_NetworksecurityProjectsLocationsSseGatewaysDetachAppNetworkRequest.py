from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityProjectsLocationsSseGatewaysDetachAppNetworkRequest(_messages.Message):
    """A NetworksecurityProjectsLocationsSseGatewaysDetachAppNetworkRequest
  object.

  Fields:
    detachAppNetworkRequest: A DetachAppNetworkRequest resource to be passed
      as the request body.
    name: Required. Name of the SSEGateway which holds the detached network.
      Must be in the format `projects/*/locations/{location}/sseGateways/*`.
  """
    detachAppNetworkRequest = _messages.MessageField('DetachAppNetworkRequest', 1)
    name = _messages.StringField(2, required=True)