from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesDatabasesSessionsExecuteBatchDmlRequest(_messages.Message):
    """A SpannerProjectsInstancesDatabasesSessionsExecuteBatchDmlRequest
  object.

  Fields:
    executeBatchDmlRequest: A ExecuteBatchDmlRequest resource to be passed as
      the request body.
    session: Required. The session in which the DML statements should be
      performed.
  """
    executeBatchDmlRequest = _messages.MessageField('ExecuteBatchDmlRequest', 1)
    session = _messages.StringField(2, required=True)