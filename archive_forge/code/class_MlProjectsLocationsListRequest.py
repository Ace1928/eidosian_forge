from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MlProjectsLocationsListRequest(_messages.Message):
    """A MlProjectsLocationsListRequest object.

  Fields:
    pageSize: Optional. The number of locations to retrieve per "page" of
      results. If there are more remaining results than this number, the
      response message will contain a valid value in the `next_page_token`
      field. The default value is 20, and the maximum page size is 100.
    pageToken: Optional. A page token to request the next page of results. You
      get the token from the `next_page_token` field of the response from the
      previous call.
    parent: Required. The name of the project for which available locations
      are to be listed (since some locations might be whitelisted for specific
      projects).
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)