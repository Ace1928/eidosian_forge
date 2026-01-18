from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RemotebuildexecutionProjectsInstancesGetRequest(_messages.Message):
    """A RemotebuildexecutionProjectsInstancesGetRequest object.

  Fields:
    name: Required. Name of the instance to retrieve. Format:
      `projects/[PROJECT_ID]/instances/[INSTANCE_ID]`.
  """
    name = _messages.StringField(1, required=True)