from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConversionWorkspaceInfo(_messages.Message):
    """A conversion workspace's version.

  Fields:
    commitId: The commit ID of the conversion workspace.
    name: The resource name (URI) of the conversion workspace.
  """
    commitId = _messages.StringField(1)
    name = _messages.StringField(2)