from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OriginHeaderAction(_messages.Message):
    """Defines the addition and removal of HTTP headers for requests/responses.

  Fields:
    requestHeadersToAdd: Optional. A header to add. You can add a maximum of
      25 request headers.
  """
    requestHeadersToAdd = _messages.MessageField('OriginHeaderActionAddHeader', 1, repeated=True)