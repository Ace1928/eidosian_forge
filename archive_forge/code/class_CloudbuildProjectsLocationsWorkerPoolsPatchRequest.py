from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsWorkerPoolsPatchRequest(_messages.Message):
    """A CloudbuildProjectsLocationsWorkerPoolsPatchRequest object.

  Fields:
    name: Output only. The resource name of the `WorkerPool`, with format
      `projects/{project}/locations/{location}/workerPools/{worker_pool}`. The
      value of `{worker_pool}` is provided by `worker_pool_id` in
      `CreateWorkerPool` request and the value of `{location}` is determined
      by the endpoint accessed.
    updateMask: A mask specifying which fields in `worker_pool` to update.
    validateOnly: If set, validate the request and preview the response, but
      do not actually post it.
    workerPool: A WorkerPool resource to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    updateMask = _messages.StringField(2)
    validateOnly = _messages.BooleanField(3)
    workerPool = _messages.MessageField('WorkerPool', 4)