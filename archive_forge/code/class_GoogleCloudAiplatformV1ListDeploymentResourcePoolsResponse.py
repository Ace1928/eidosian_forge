from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ListDeploymentResourcePoolsResponse(_messages.Message):
    """Response message for ListDeploymentResourcePools method.

  Fields:
    deploymentResourcePools: The DeploymentResourcePools from the specified
      location.
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
  """
    deploymentResourcePools = _messages.MessageField('GoogleCloudAiplatformV1DeploymentResourcePool', 1, repeated=True)
    nextPageToken = _messages.StringField(2)