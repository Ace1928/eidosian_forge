from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkspaceDeclaration(_messages.Message):
    """WorkspaceDeclaration is a declaration of a volume that a Task requires.

  Fields:
    description: Description is a human readable description of this volume.
    mountPath: MountPath overrides the directory that the volume will be made
      available at.
    name: Name is the name by which you can bind the volume at runtime.
    optional: Optional. Optional marks a Workspace as not being required in
      TaskRuns. By default this field is false and so declared workspaces are
      required.
    readOnly: ReadOnly dictates whether a mounted volume is writable.
  """
    description = _messages.StringField(1)
    mountPath = _messages.StringField(2)
    name = _messages.StringField(3)
    optional = _messages.BooleanField(4)
    readOnly = _messages.BooleanField(5)