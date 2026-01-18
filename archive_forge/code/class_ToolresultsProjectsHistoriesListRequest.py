from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ToolresultsProjectsHistoriesListRequest(_messages.Message):
    """A ToolresultsProjectsHistoriesListRequest object.

  Fields:
    filterByName: If set, only return histories with the given name. Optional.
    pageSize: The maximum number of Histories to fetch. Default value: 20. The
      server will use this default if the field is not set or has a value of
      0. Any value greater than 100 will be treated as 100. Optional.
    pageToken: A continuation token to resume the query at the next item.
      Optional.
    projectId: A Project id. Required.
  """
    filterByName = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    projectId = _messages.StringField(4, required=True)