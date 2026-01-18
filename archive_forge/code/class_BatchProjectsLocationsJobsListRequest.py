from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchProjectsLocationsJobsListRequest(_messages.Message):
    """A BatchProjectsLocationsJobsListRequest object.

  Fields:
    filter: List filter.
    orderBy: Optional. Sort results. Supported are "name", "name desc",
      "create_time", and "create_time desc".
    pageSize: Page size.
    pageToken: Page token.
    parent: Parent path.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)