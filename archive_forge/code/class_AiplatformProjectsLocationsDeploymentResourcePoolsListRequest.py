from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsDeploymentResourcePoolsListRequest(_messages.Message):
    """A AiplatformProjectsLocationsDeploymentResourcePoolsListRequest object.

  Fields:
    pageSize: The maximum number of DeploymentResourcePools to return. The
      service may return fewer than this value.
    pageToken: A page token, received from a previous
      `ListDeploymentResourcePools` call. Provide this to retrieve the
      subsequent page. When paginating, all other parameters provided to
      `ListDeploymentResourcePools` must match the call that provided the page
      token.
    parent: Required. The parent Location which owns this collection of
      DeploymentResourcePools. Format:
      `projects/{project}/locations/{location}`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)