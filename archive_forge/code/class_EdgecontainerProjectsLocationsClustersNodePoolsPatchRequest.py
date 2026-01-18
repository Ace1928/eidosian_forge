from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EdgecontainerProjectsLocationsClustersNodePoolsPatchRequest(_messages.Message):
    """A EdgecontainerProjectsLocationsClustersNodePoolsPatchRequest object.

  Fields:
    name: Required. The resource name of the node pool.
    nodePool: A NodePool resource to be passed as the request body.
    requestId: A unique identifier for this request. Restricted to 36 ASCII
      characters. A random UUID is recommended. This request is only
      idempotent if `request_id` is provided.
    updateMask: Field mask is used to specify the fields to be overwritten in
      the NodePool resource by the update. The fields specified in the
      update_mask are relative to the resource, not the full request. A field
      will be overwritten if it is in the mask. If the user does not provide a
      mask then all fields will be overwritten.
  """
    name = _messages.StringField(1, required=True)
    nodePool = _messages.MessageField('NodePool', 2)
    requestId = _messages.StringField(3)
    updateMask = _messages.StringField(4)