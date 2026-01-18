from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeonpremProjectsLocationsBareMetalStandaloneClustersUnenrollRequest(_messages.Message):
    """A GkeonpremProjectsLocationsBareMetalStandaloneClustersUnenrollRequest
  object.

  Fields:
    allowMissing: If set to true, and the bare metal standalone cluster is not
      found, the request will succeed but no action will be taken on the
      server and return a completed LRO.
    etag: The current etag of the bare metal standalone cluster. If an etag is
      provided and does not match the current etag of the cluster, deletion
      will be blocked and an ABORTED error will be returned.
    force: This is required if the cluster has any associated node pools. When
      set, any child node pools will also be unenrolled.
    ignoreErrors: Optional. If set to true, the unenrollment of a bare metal
      standalone cluster resource will succeed even if errors occur during
      unenrollment. This parameter can be used when you want to unenroll
      standalone cluster resource and the on-prem standalone cluster is
      disconnected / unreachable. WARNING: Using this parameter when your
      standalone cluster still exists may result in a deleted GCP standalone
      cluster but existing resourcelink in on-prem standalone cluster and
      membership.
    name: Required. Name of the bare metal standalone cluster to be
      unenrolled. Format: "projects/{project}/locations/{location}/bareMetalSt
      andaloneClusters/{cluster}"
    validateOnly: Validate the request without actually doing any updates.
  """
    allowMissing = _messages.BooleanField(1)
    etag = _messages.StringField(2)
    force = _messages.BooleanField(3)
    ignoreErrors = _messages.BooleanField(4)
    name = _messages.StringField(5, required=True)
    validateOnly = _messages.BooleanField(6)