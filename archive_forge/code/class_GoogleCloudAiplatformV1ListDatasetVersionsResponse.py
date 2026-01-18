from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ListDatasetVersionsResponse(_messages.Message):
    """Response message for DatasetService.ListDatasetVersions.

  Fields:
    datasetVersions: A list of DatasetVersions that matches the specified
      filter in the request.
    nextPageToken: The standard List next-page token.
  """
    datasetVersions = _messages.MessageField('GoogleCloudAiplatformV1DatasetVersion', 1, repeated=True)
    nextPageToken = _messages.StringField(2)