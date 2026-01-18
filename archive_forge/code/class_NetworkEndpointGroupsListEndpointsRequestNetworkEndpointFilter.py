from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkEndpointGroupsListEndpointsRequestNetworkEndpointFilter(_messages.Message):
    """A NetworkEndpointGroupsListEndpointsRequestNetworkEndpointFilter object.

  Fields:
    networkEndpoint: A NetworkEndpoint attribute.
  """
    networkEndpoint = _messages.MessageField('NetworkEndpoint', 1)