from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GsuiteaddonsProjectsDeploymentsListRequest(_messages.Message):
    """A GsuiteaddonsProjectsDeploymentsListRequest object.

  Fields:
    pageSize: The maximum number of deployments to return. The service might
      return fewer than this value. If unspecified, at most 1,000 deployments
      are returned. The maximum possible value is 1,000; values above 1,000
      are changed to 1,000.
    pageToken: A page token, received from a previous `ListDeployments` call.
      Provide this to retrieve the subsequent page. When paginating, all other
      parameters provided to `ListDeployments` must match the call that
      provided the page token.
    parent: Required. Name of the project in which to create the deployment.
      Example: `projects/my_project`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)