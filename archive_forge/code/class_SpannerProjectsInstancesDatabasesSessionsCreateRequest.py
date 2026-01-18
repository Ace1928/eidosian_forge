from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesDatabasesSessionsCreateRequest(_messages.Message):
    """A SpannerProjectsInstancesDatabasesSessionsCreateRequest object.

  Fields:
    createSessionRequest: A CreateSessionRequest resource to be passed as the
      request body.
    database: Required. The database in which the new session is created.
  """
    createSessionRequest = _messages.MessageField('CreateSessionRequest', 1)
    database = _messages.StringField(2, required=True)