from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsWorkerPoolsDeleteRequest(_messages.Message):
    """A CloudbuildProjectsWorkerPoolsDeleteRequest object.

  Fields:
    name: Required. The name of the `WorkerPool` to delete. Format:
      projects/{project}/workerPools/{workerPool}
  """
    name = _messages.StringField(1, required=True)