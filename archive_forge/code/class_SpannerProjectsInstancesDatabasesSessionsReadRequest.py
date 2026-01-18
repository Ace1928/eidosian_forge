from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesDatabasesSessionsReadRequest(_messages.Message):
    """A SpannerProjectsInstancesDatabasesSessionsReadRequest object.

  Fields:
    readRequest: A ReadRequest resource to be passed as the request body.
    session: Required. The session in which the read should be performed.
  """
    readRequest = _messages.MessageField('ReadRequest', 1)
    session = _messages.StringField(2, required=True)