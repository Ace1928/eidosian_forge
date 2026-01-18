from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeonpremProjectsLocationsVmwareAdminClustersUnenrollRequest(_messages.Message):
    """A GkeonpremProjectsLocationsVmwareAdminClustersUnenrollRequest object.

  Fields:
    allowMissing: If set to true, and the VMware admin cluster is not found,
      the request will succeed but no action will be taken on the server and
      return a completed LRO.
    etag: The current etag of the VMware admin cluster. If an etag is provided
      and does not match the current etag of the cluster, deletion will be
      blocked and an ABORTED error will be returned.
    name: Required. Name of the VMware admin cluster to be unenrolled. Format:
      "projects/{project}/locations/{location}/vmwareAdminClusters/{cluster}"
    validateOnly: Validate the request without actually doing any updates.
  """
    allowMissing = _messages.BooleanField(1)
    etag = _messages.StringField(2)
    name = _messages.StringField(3, required=True)
    validateOnly = _messages.BooleanField(4)