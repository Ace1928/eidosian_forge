from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeGlobalNetworkEndpointGroupsGetRequest(_messages.Message):
    """A ComputeGlobalNetworkEndpointGroupsGetRequest object.

  Fields:
    networkEndpointGroup: The name of the network endpoint group. It should
      comply with RFC1035.
    project: Project ID for this request.
  """
    networkEndpointGroup = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)