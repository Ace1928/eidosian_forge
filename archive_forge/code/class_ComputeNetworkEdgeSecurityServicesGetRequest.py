from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeNetworkEdgeSecurityServicesGetRequest(_messages.Message):
    """A ComputeNetworkEdgeSecurityServicesGetRequest object.

  Fields:
    networkEdgeSecurityService: Name of the network edge security service to
      get.
    project: Project ID for this request.
    region: Name of the region scoping this request.
  """
    networkEdgeSecurityService = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)
    region = _messages.StringField(3, required=True)