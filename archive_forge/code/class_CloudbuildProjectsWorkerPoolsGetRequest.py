from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsWorkerPoolsGetRequest(_messages.Message):
    """A CloudbuildProjectsWorkerPoolsGetRequest object.

  Fields:
    name: Required. The name of the `WorkerPool` to retrieve. Format:
      projects/{project}/workerPools/{workerPool}
  """
    name = _messages.StringField(1, required=True)