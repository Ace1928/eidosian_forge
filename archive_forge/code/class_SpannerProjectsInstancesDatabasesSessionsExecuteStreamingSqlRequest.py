from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesDatabasesSessionsExecuteStreamingSqlRequest(_messages.Message):
    """A SpannerProjectsInstancesDatabasesSessionsExecuteStreamingSqlRequest
  object.

  Fields:
    executeSqlRequest: A ExecuteSqlRequest resource to be passed as the
      request body.
    session: Required. The session in which the SQL query should be performed.
  """
    executeSqlRequest = _messages.MessageField('ExecuteSqlRequest', 1)
    session = _messages.StringField(2, required=True)