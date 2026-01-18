from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowProjectsLocationsDeploymentsListRevisionsRequest(_messages.Message):
    """A DataflowProjectsLocationsDeploymentsListRevisionsRequest object.

  Fields:
    name: Required. The name of the `Deployment`. Format:
      projects/{project}/locations/{location}/deployments/{deployment_id}
    pageSize: The maximum number of deployments to return. The service may
      return fewer than this value. If unspecified, at most 50 deployments
      will be returned. The maximum value is 1000; values above 1000 will be
      coerced to 1000.
    pageToken: A page token, received from a previous `ListDeployments` call.
      Provide this to retrieve the subsequent page. When paginating, all other
      parameters provided to `ListDeployments` must match the call that
      provided the page token.
  """
    name = _messages.StringField(1, required=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)