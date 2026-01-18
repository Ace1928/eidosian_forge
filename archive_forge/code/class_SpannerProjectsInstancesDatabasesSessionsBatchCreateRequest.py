from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesDatabasesSessionsBatchCreateRequest(_messages.Message):
    """A SpannerProjectsInstancesDatabasesSessionsBatchCreateRequest object.

  Fields:
    batchCreateSessionsRequest: A BatchCreateSessionsRequest resource to be
      passed as the request body.
    database: Required. The database in which the new sessions are created.
  """
    batchCreateSessionsRequest = _messages.MessageField('BatchCreateSessionsRequest', 1)
    database = _messages.StringField(2, required=True)