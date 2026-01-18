from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ToolresultsProjectsInitializeSettingsRequest(_messages.Message):
    """A ToolresultsProjectsInitializeSettingsRequest object.

  Fields:
    projectId: A Project id. Required.
  """
    projectId = _messages.StringField(1, required=True)