from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ListArchiveDeploymentsResponse(_messages.Message):
    """Response for ListArchiveDeployments method.

  Fields:
    archiveDeployments: Archive Deployments in the specified environment.
    nextPageToken: Page token that you can include in a ListArchiveDeployments
      request to retrieve the next page. If omitted, no subsequent pages
      exist.
  """
    archiveDeployments = _messages.MessageField('GoogleCloudApigeeV1ArchiveDeployment', 1, repeated=True)
    nextPageToken = _messages.StringField(2)