from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RemotebuildexecutionProjectsInstancesWorkerpoolsGetRequest(_messages.Message):
    """A RemotebuildexecutionProjectsInstancesWorkerpoolsGetRequest object.

  Fields:
    name: Name of the worker pool to retrieve. Format:
      `projects/[PROJECT_ID]/instances/[INSTANCE_ID]/workerpools/[POOL_ID]`.
  """
    name = _messages.StringField(1, required=True)