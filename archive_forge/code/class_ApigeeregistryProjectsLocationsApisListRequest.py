from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeregistryProjectsLocationsApisListRequest(_messages.Message):
    """A ApigeeregistryProjectsLocationsApisListRequest object.

  Fields:
    filter: An expression that can be used to filter the list. Filters use the
      Common Expression Language and can refer to all message fields.
    orderBy: A comma-separated list of fields, e.g. "foo,bar" Fields can be
      sorted in descending order using the "desc" identifier, e.g. "foo
      desc,bar"
    pageSize: The maximum number of APIs to return. The service may return
      fewer than this value. If unspecified, at most 50 values will be
      returned. The maximum is 1000; values above 1000 will be coerced to
      1000.
    pageToken: A page token, received from a previous `ListApis` call. Provide
      this to retrieve the subsequent page. When paginating, all other
      parameters provided to `ListApis` must match the call that provided the
      page token.
    parent: Required. The parent, which owns this collection of APIs. Format:
      `projects/*/locations/*`
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)