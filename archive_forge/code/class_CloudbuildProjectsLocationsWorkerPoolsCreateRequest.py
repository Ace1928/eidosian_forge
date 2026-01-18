from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsWorkerPoolsCreateRequest(_messages.Message):
    """A CloudbuildProjectsLocationsWorkerPoolsCreateRequest object.

  Fields:
    parent: Required. The parent resource where this worker pool will be
      created. Format: `projects/{project}/locations/{location}`.
    validateOnly: If set, validate the request and preview the response, but
      do not actually post it.
    workerPool: A WorkerPool resource to be passed as the request body.
    workerPoolId: Required. Immutable. The ID to use for the `WorkerPool`,
      which will become the final component of the resource name. This value
      should be 1-63 characters, and valid characters are /a-z-/.
  """
    parent = _messages.StringField(1, required=True)
    validateOnly = _messages.BooleanField(2)
    workerPool = _messages.MessageField('WorkerPool', 3)
    workerPoolId = _messages.StringField(4)