from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CancelBuildRequest(_messages.Message):
    """Request to cancel an ongoing build.

  Fields:
    id: Required. ID of the build.
    name: The name of the `Build` to cancel. Format:
      `projects/{project}/locations/{location}/builds/{build}`
    projectId: Required. ID of the project.
  """
    id = _messages.StringField(1)
    name = _messages.StringField(2)
    projectId = _messages.StringField(3)