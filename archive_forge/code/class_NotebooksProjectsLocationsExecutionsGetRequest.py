from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotebooksProjectsLocationsExecutionsGetRequest(_messages.Message):
    """A NotebooksProjectsLocationsExecutionsGetRequest object.

  Fields:
    name: Required. Format:
      `projects/{project_id}/locations/{location}/executions/{execution_id}`
  """
    name = _messages.StringField(1, required=True)