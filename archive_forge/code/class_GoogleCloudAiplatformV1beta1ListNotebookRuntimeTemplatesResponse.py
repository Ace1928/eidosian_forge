from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ListNotebookRuntimeTemplatesResponse(_messages.Message):
    """Response message for NotebookService.ListNotebookRuntimeTemplates.

  Fields:
    nextPageToken: A token to retrieve next page of results. Pass to
      ListNotebookRuntimeTemplatesRequest.page_token to obtain that page.
    notebookRuntimeTemplates: List of NotebookRuntimeTemplates in the
      requested page.
  """
    nextPageToken = _messages.StringField(1)
    notebookRuntimeTemplates = _messages.MessageField('GoogleCloudAiplatformV1beta1NotebookRuntimeTemplate', 2, repeated=True)