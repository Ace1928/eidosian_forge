from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TpuProjectsLocationsQueuedResourcesListRequest(_messages.Message):
    """A TpuProjectsLocationsQueuedResourcesListRequest object.

  Fields:
    pageSize: Optional. The maximum number of items to return.
    pageToken: Optional. The next_page_token value returned from a previous
      List request, if any.
    parent: Required. The parent resource name.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)