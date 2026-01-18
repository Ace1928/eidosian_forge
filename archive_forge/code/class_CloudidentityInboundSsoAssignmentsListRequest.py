from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityInboundSsoAssignmentsListRequest(_messages.Message):
    """A CloudidentityInboundSsoAssignmentsListRequest object.

  Fields:
    filter: A CEL expression to filter the results. The only supported filter
      is filtering by customer. For example: `customer==customers/C0123abc`.
      Omitting the filter or specifying a filter of
      `customer==customers/my_customer` will return the assignments for the
      customer that the caller (authenticated user) belongs to.
    pageSize: The maximum number of assignments to return. The service may
      return fewer than this value. If omitted (or defaulted to zero) the
      server will use a sensible default. This default may change over time.
      The maximum allowed value is 100, though requests with page_size greater
      than that will be silently interpreted as having this maximum value.
      This may increase in the futue.
    pageToken: A page token, received from a previous
      `ListInboundSsoAssignments` call. Provide this to retrieve the
      subsequent page. When paginating, all other parameters provided to
      `ListInboundSsoAssignments` must match the call that provided the page
      token.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)