from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeInstanceGroupManagerResizeRequestsGetRequest(_messages.Message):
    """A ComputeInstanceGroupManagerResizeRequestsGetRequest object.

  Fields:
    instanceGroupManager: The name of the managed instance group. Name should
      conform to RFC1035 or be a resource ID.
    project: Project ID for this request.
    resizeRequest: The name of the resize request. Name should conform to
      RFC1035 or be a resource ID.
    zone: Name of the href="/compute/docs/regions-zones/#available">zone
      scoping this request. Name should conform to RFC1035.
  """
    instanceGroupManager = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)
    resizeRequest = _messages.StringField(3, required=True)
    zone = _messages.StringField(4, required=True)