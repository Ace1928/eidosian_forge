from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsWorkerPoolsDeleteRequest(_messages.Message):
    """A CloudbuildProjectsLocationsWorkerPoolsDeleteRequest object.

  Fields:
    allowMissing: If set to true, and the `WorkerPool` is not found, the
      request will succeed but no action will be taken on the server.
    etag: Optional. If provided, it must match the server's etag on the
      workerpool for the request to be processed.
    name: Required. The name of the `WorkerPool` to delete. Format:
      `projects/{project}/locations/{location}/workerPools/{workerPool}`.
    validateOnly: If set, validate the request and preview the response, but
      do not actually post it.
  """
    allowMissing = _messages.BooleanField(1)
    etag = _messages.StringField(2)
    name = _messages.StringField(3, required=True)
    validateOnly = _messages.BooleanField(4)