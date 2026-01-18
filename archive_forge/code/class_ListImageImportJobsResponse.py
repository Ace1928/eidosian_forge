from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListImageImportJobsResponse(_messages.Message):
    """Response message for 'ListImageImportJobs' call.

  Fields:
    imageImportJobs: Output only. The list of target response.
    nextPageToken: Output only. A token, which can be sent as `page_token` to
      retrieve the next page. If this field is omitted, there are no
      subsequent pages.
    unreachable: Output only. Locations that could not be reached.
  """
    imageImportJobs = _messages.MessageField('ImageImportJob', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)