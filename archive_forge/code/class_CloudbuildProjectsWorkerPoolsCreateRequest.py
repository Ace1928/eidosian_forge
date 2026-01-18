from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsWorkerPoolsCreateRequest(_messages.Message):
    """A CloudbuildProjectsWorkerPoolsCreateRequest object.

  Fields:
    parent: Required. The parent resource where this book will be created.
      Format: projects/{project}
    workerPool: A WorkerPool resource to be passed as the request body.
    workerPoolId: Required. Immutable. The ID to use for the `WorkerPool`,
      which will become the final component of the resource name. This value
      should be 1-63 characters, and valid characters are /a-z-/.
  """
    parent = _messages.StringField(1, required=True)
    workerPool = _messages.MessageField('WorkerPool', 2)
    workerPoolId = _messages.StringField(3)