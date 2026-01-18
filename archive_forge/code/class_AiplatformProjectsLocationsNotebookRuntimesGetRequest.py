from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsNotebookRuntimesGetRequest(_messages.Message):
    """A AiplatformProjectsLocationsNotebookRuntimesGetRequest object.

  Fields:
    name: Required. The name of the NotebookRuntime resource. Instead of
      checking whether the name is in valid NotebookRuntime resource name
      format, directly throw NotFound exception if there is no such
      NotebookRuntime in spanner.
  """
    name = _messages.StringField(1, required=True)