from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsNotebookRuntimeTemplatesGetRequest(_messages.Message):
    """A AiplatformProjectsLocationsNotebookRuntimeTemplatesGetRequest object.

  Fields:
    name: Required. The name of the NotebookRuntimeTemplate resource. Format:
      `projects/{project}/locations/{location}/notebookRuntimeTemplates/{noteb
      ook_runtime_template}`
  """
    name = _messages.StringField(1, required=True)