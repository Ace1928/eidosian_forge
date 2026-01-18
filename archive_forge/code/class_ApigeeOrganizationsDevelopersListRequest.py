from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsDevelopersListRequest(_messages.Message):
    """A ApigeeOrganizationsDevelopersListRequest object.

  Fields:
    app: Optional. List only Developers that are associated with the app. Note
      that start_key, count are not applicable for this filter criteria.
    count: Optional. Number of developers to return in the API call. Use with
      the `startKey` parameter to provide more targeted filtering. The limit
      is 1000.
    expand: Specifies whether to expand the results. Set to `true` to expand
      the results. This query parameter is not valid if you use the `count` or
      `startKey` query parameters.
    filter: Optional. The filter expression to be used to get the list of
      developers, where filtering can be done on email. Example: filter =
      "email = foo@bar.com"
    ids: Optional. List of IDs to include, separated by commas.
    includeCompany: Flag that specifies whether to include company details in
      the response.
    pageSize: Optional. Count of developers a single page can have in the
      response. If unspecified, at most 100 developers will be returned. The
      maximum value is 100; values above 100 will be coerced to 100.
    pageToken: Optional. The starting index record for listing the developers.
    parent: Required. Name of the Apigee organization. Use the following
      structure in your request: `organizations/{org}`.
    startKey: **Note**: Must be used in conjunction with the `count`
      parameter. Email address of the developer from which to start displaying
      the list of developers. For example, if the an unfiltered list returns:
      ``` westley@example.com fezzik@example.com buttercup@example.com ``` and
      your `startKey` is `fezzik@example.com`, the list returned will be ```
      fezzik@example.com buttercup@example.com ```
  """
    app = _messages.StringField(1)
    count = _messages.IntegerField(2)
    expand = _messages.BooleanField(3)
    filter = _messages.StringField(4)
    ids = _messages.StringField(5)
    includeCompany = _messages.BooleanField(6)
    pageSize = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(8)
    parent = _messages.StringField(9, required=True)
    startKey = _messages.StringField(10)