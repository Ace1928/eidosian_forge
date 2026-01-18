from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsDatacollectorsListRequest(_messages.Message):
    """A ApigeeOrganizationsDatacollectorsListRequest object.

  Fields:
    pageSize: Maximum number of data collectors to return. The page size
      defaults to 25.
    pageToken: Page token, returned from a previous ListDataCollectors call,
      that you can use to retrieve the next page.
    parent: Required. Name of the organization for which to list data
      collectors in the following format: `organizations/{org}`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)