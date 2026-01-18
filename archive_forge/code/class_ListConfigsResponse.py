from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListConfigsResponse(_messages.Message):
    """`ListConfigs()` returns the following response. The order of returned
  objects is arbitrary; that is, it is not ordered in any particular way.

  Fields:
    configs: A list of the configurations in the project. The order of
      returned objects is arbitrary; that is, it is not ordered in any
      particular way.
    nextPageToken: This token allows you to get the next page of results for
      list requests. If the number of results is larger than `pageSize`, use
      the `nextPageToken` as a value for the query parameter `pageToken` in
      the next list request. Subsequent list requests will have their own
      `nextPageToken` to continue paging through the results
  """
    configs = _messages.MessageField('RuntimeConfig', 1, repeated=True)
    nextPageToken = _messages.StringField(2)