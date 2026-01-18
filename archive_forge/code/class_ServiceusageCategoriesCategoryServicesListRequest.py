from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceusageCategoriesCategoryServicesListRequest(_messages.Message):
    """A ServiceusageCategoriesCategoryServicesListRequest object.

  Fields:
    pageSize: The maximum number of services to return. The service may return
      fewer than this value. If unspecified, at most 50 groups will be
      returned. The maximum value is 1000; values above 1000 will be coerced
      to 1000.
    pageToken: A page token, received from a previous `ListCategoryServices`
      call. Provide this to retrieve the subsequent page. When paginating, all
      other parameters provided to `ListCategoryServices` must match the call
      that provided the page token.
    parent: Required. The category which contains the returned services.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)