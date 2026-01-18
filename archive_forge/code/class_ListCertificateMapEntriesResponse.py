from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListCertificateMapEntriesResponse(_messages.Message):
    """Response for the `ListCertificateMapEntries` method.

  Fields:
    certificateMapEntries: A list of certificate map entries for the parent
      resource.
    nextPageToken: If there might be more results than those appearing in this
      response, then `next_page_token` is included. To get the next set of
      results, call this method again using the value of `next_page_token` as
      `page_token`.
    unreachable: Locations that could not be reached.
  """
    certificateMapEntries = _messages.MessageField('CertificateMapEntry', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)