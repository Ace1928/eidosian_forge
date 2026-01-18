from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkspacePipelineTaskBinding(_messages.Message):
    """WorkspacePipelineTaskBinding maps workspaces from the PipelineSpec to
  the workspaces declared in the Task.

  Fields:
    name: Name of the workspace as declared by the task.
    subPath: Optional. SubPath is optionally a directory on the volume which
      should be used for this binding (i.e. the volume will be mounted at this
      sub directory). +optional
    workspace: Name of the workspace declared by the pipeline.
  """
    name = _messages.StringField(1)
    subPath = _messages.StringField(2)
    workspace = _messages.StringField(3)