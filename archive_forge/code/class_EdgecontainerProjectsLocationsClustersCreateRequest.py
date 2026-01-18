from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EdgecontainerProjectsLocationsClustersCreateRequest(_messages.Message):
    """A EdgecontainerProjectsLocationsClustersCreateRequest object.

  Fields:
    cluster: A Cluster resource to be passed as the request body.
    clusterId: Required. A client-specified unique identifier for the cluster.
    parent: Required. The parent location where this cluster will be created.
    requestId: A unique identifier for this request. Restricted to 36 ASCII
      characters. A random UUID is recommended. This request is only
      idempotent if `request_id` is provided.
  """
    cluster = _messages.MessageField('Cluster', 1)
    clusterId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    requestId = _messages.StringField(4)