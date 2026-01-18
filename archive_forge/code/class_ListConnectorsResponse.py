from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListConnectorsResponse(_messages.Message):
    """Response for listing Serverless VPC Access connectors.

  Fields:
    connectors: List of Serverless VPC Access connectors.
    nextPageToken: Continuation token.
  """
    connectors = _messages.MessageField('Connector', 1, repeated=True)
    nextPageToken = _messages.StringField(2)