from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranscoderProjectsLocationsJobTemplatesListRequest(_messages.Message):
    """A TranscoderProjectsLocationsJobTemplatesListRequest object.

  Fields:
    filter: The filter expression, following the syntax outlined in
      https://google.aip.dev/160.
    orderBy: One or more fields to compare and use to sort the output. See
      https://google.aip.dev/132#ordering.
    pageSize: The maximum number of items to return.
    pageToken: The `next_page_token` value returned from a previous List
      request, if any.
    parent: Required. The parent location from which to retrieve the
      collection of job templates. Format:
      `projects/{project}/locations/{location}`
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)