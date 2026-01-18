from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeregistryProjectsLocationsApisDeploymentsListRevisionsRequest(_messages.Message):
    """A ApigeeregistryProjectsLocationsApisDeploymentsListRevisionsRequest
  object.

  Fields:
    filter: An expression that can be used to filter the list. Filters use the
      Common Expression Language and can refer to all message fields.
    name: Required. The name of the deployment to list revisions for.
    pageSize: The maximum number of revisions to return per page.
    pageToken: The page token, received from a previous
      ListApiDeploymentRevisions call. Provide this to retrieve the subsequent
      page.
  """
    filter = _messages.StringField(1)
    name = _messages.StringField(2, required=True)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)