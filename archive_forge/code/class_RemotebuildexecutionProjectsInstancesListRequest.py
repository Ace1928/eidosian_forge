from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RemotebuildexecutionProjectsInstancesListRequest(_messages.Message):
    """A RemotebuildexecutionProjectsInstancesListRequest object.

  Fields:
    parent: Required. Resource name of the project. Format:
      `projects/[PROJECT_ID]`.
  """
    parent = _messages.StringField(1, required=True)