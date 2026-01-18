from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotebooksProjectsLocationsInstancesDiagnoseRequest(_messages.Message):
    """A NotebooksProjectsLocationsInstancesDiagnoseRequest object.

  Fields:
    diagnoseInstanceRequest: A DiagnoseInstanceRequest resource to be passed
      as the request body.
    name: Required. Format:
      `projects/{project_id}/locations/{location}/instances/{instance_id}`
  """
    diagnoseInstanceRequest = _messages.MessageField('DiagnoseInstanceRequest', 1)
    name = _messages.StringField(2, required=True)