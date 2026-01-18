from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeonpremProjectsLocationsVmwareClustersVmwareNodePoolsDeleteRequest(_messages.Message):
    """A GkeonpremProjectsLocationsVmwareClustersVmwareNodePoolsDeleteRequest
  object.

  Fields:
    allowMissing: If set to true, and the VMware node pool is not found, the
      request will succeed but no action will be taken on the server and
      return a completed LRO.
    etag: The current etag of the VmwareNodePool. If an etag is provided and
      does not match the current etag of the node pool, deletion will be
      blocked and an ABORTED error will be returned.
    ignoreErrors: If set to true, the deletion of a VMware node pool resource
      will succeed even if errors occur during deletion. This parameter can be
      used when you want to delete GCP's node pool resource and you've already
      deleted the on-prem admin cluster that hosted your node pool. WARNING:
      Using this parameter when your user cluster still exists may result in a
      deleted GCP node pool but an existing on-prem node pool.
    name: Required. The name of the node pool to delete. Format: projects/{pro
      ject}/locations/{location}/vmwareClusters/{cluster}/vmwareNodePools/{nod
      epool}
    validateOnly: If set, only validate the request, but do not actually
      delete the node pool.
  """
    allowMissing = _messages.BooleanField(1)
    etag = _messages.StringField(2)
    ignoreErrors = _messages.BooleanField(3)
    name = _messages.StringField(4, required=True)
    validateOnly = _messages.BooleanField(5)