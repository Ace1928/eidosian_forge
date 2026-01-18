from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListPatchDeploymentsResponse(_messages.Message):
    """A response message for listing patch deployments.

  Fields:
    nextPageToken: A pagination token that can be used to get the next page of
      patch deployments.
    patchDeployments: The list of patch deployments.
  """
    nextPageToken = _messages.StringField(1)
    patchDeployments = _messages.MessageField('PatchDeployment', 2, repeated=True)