from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeonpremProjectsLocationsBareMetalClustersBareMetalNodePoolsUnenrollRequest(_messages.Message):
    """A
  GkeonpremProjectsLocationsBareMetalClustersBareMetalNodePoolsUnenrollRequest
  object.

  Fields:
    allowMissing: If set to true, and the bare metal node pool is not found,
      the request will succeed but no action will be taken on the server and
      return a completed LRO.
    etag: The current etag of the bare metal node pool. If an etag is provided
      and does not match the current etag of node pool, deletion will be
      blocked and an ABORTED error will be returned.
    name: Required. The name of the node pool to unenroll. Format: projects/{p
      roject}/locations/{location}/bareMetalClusters/{cluster}/bareMetalNodePo
      ols/{nodepool}
    validateOnly: If set, only validate the request, but do not actually
      unenroll the node pool.
  """
    allowMissing = _messages.BooleanField(1)
    etag = _messages.StringField(2)
    name = _messages.StringField(3, required=True)
    validateOnly = _messages.BooleanField(4)