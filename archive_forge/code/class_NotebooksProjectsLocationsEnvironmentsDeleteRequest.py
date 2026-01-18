from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotebooksProjectsLocationsEnvironmentsDeleteRequest(_messages.Message):
    """A NotebooksProjectsLocationsEnvironmentsDeleteRequest object.

  Fields:
    name: Required. Format:
      `projects/{project_id}/locations/{location}/environments/{environment_id
      }`
  """
    name = _messages.StringField(1, required=True)