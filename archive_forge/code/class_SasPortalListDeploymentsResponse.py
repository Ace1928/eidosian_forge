from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalListDeploymentsResponse(_messages.Message):
    """Response for ListDeployments.

  Fields:
    deployments: The deployments that match the request.
    nextPageToken: A pagination token returned from a previous call to
      ListDeployments that indicates from where listing should continue. If
      the field is missing or empty, it means there are no more deployments.
  """
    deployments = _messages.MessageField('SasPortalDeployment', 1, repeated=True)
    nextPageToken = _messages.StringField(2)