from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsWorkerPoolsListRequest(_messages.Message):
    """A CloudbuildProjectsWorkerPoolsListRequest object.

  Fields:
    parent: Required. The parent, which owns this collection of `WorkerPools`.
      Format: projects/{project}
  """
    parent = _messages.StringField(1, required=True)