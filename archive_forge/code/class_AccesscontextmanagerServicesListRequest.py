from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccesscontextmanagerServicesListRequest(_messages.Message):
    """A AccesscontextmanagerServicesListRequest object.

  Fields:
    pageSize: This flag specifies the maximum number of services to return per
      page. Default is 100.
    pageToken: Token to start on a later page. Default is the first page.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)