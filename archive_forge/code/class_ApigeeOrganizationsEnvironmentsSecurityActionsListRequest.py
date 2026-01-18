from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsSecurityActionsListRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsSecurityActionsListRequest object.

  Fields:
    filter: The filter expression to filter List results.
      https://google.aip.dev/160. Allows for filtering over: state and
      api_proxies. E.g.: state = ACTIVE AND apiProxies:foo. Filtering by
      action is not supported https://github.com/aip-
      dev/google.aip.dev/issues/624
    pageSize: The maximum number of SecurityActions to return. If unspecified,
      at most 50 SecurityActions will be returned. The maximum value is 1000;
      values above 1000 will be coerced to 1000.
    pageToken: A page token, received from a previous `ListSecurityActions`
      call. Provide this to retrieve the subsequent page. When paginating, all
      other parameters provided to `ListSecurityActions` must match the call
      that provided the page token.
    parent: Required. The parent, which owns this collection of
      SecurityActions. Format: organizations/{org}/environments/{env}
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)