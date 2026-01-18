from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeSubnetworksGetRequest(_messages.Message):
    """A ComputeSubnetworksGetRequest object.

  Fields:
    project: Project ID for this request.
    region: Name of the region scoping this request.
    subnetwork: Name of the Subnetwork resource to return.
  """
    project = _messages.StringField(1, required=True)
    region = _messages.StringField(2, required=True)
    subnetwork = _messages.StringField(3, required=True)