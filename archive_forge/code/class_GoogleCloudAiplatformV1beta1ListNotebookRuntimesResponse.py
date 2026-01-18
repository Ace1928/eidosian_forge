from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ListNotebookRuntimesResponse(_messages.Message):
    """Response message for NotebookService.ListNotebookRuntimes.

  Fields:
    nextPageToken: A token to retrieve next page of results. Pass to
      ListNotebookRuntimesRequest.page_token to obtain that page.
    notebookRuntimes: List of NotebookRuntimes in the requested page.
  """
    nextPageToken = _messages.StringField(1)
    notebookRuntimes = _messages.MessageField('GoogleCloudAiplatformV1beta1NotebookRuntime', 2, repeated=True)