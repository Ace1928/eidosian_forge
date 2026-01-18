from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotebooksProjectsLocationsRuntimesDiagnoseRequest(_messages.Message):
    """A NotebooksProjectsLocationsRuntimesDiagnoseRequest object.

  Fields:
    diagnoseRuntimeRequest: A DiagnoseRuntimeRequest resource to be passed as
      the request body.
    name: Required. Format:
      `projects/{project_id}/locations/{location}/runtimes/{runtimes_id}`
  """
    diagnoseRuntimeRequest = _messages.MessageField('DiagnoseRuntimeRequest', 1)
    name = _messages.StringField(2, required=True)