from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListPerfSamplesResponse(_messages.Message):
    """A ListPerfSamplesResponse object.

  Fields:
    nextPageToken: Optional, returned if result size exceeds the page size
      specified in the request (or the default page size, 500, if
      unspecified). It indicates the last sample timestamp to be used as
      page_token in subsequent request
    perfSamples: A PerfSample attribute.
  """
    nextPageToken = _messages.StringField(1)
    perfSamples = _messages.MessageField('PerfSample', 2, repeated=True)