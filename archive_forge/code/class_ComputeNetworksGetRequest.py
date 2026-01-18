from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeNetworksGetRequest(_messages.Message):
    """A ComputeNetworksGetRequest object.

  Fields:
    network: Name of the network to return.
    project: Project ID for this request.
  """
    network = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)