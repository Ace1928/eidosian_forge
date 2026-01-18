from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsWorkerPoolsPatchRequest(_messages.Message):
    """A CloudbuildProjectsWorkerPoolsPatchRequest object.

  Fields:
    name: Output only. The resource name of the `WorkerPool`. Format of the
      name is `projects/{project_id}/workerPools/{worker_pool_id}`, where the
      value of {worker_pool_id} is provided in the CreateWorkerPool request.
    updateMask: A mask specifying which fields in `WorkerPool` should be
      updated.
    workerPool: A WorkerPool resource to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    updateMask = _messages.StringField(2)
    workerPool = _messages.MessageField('WorkerPool', 3)