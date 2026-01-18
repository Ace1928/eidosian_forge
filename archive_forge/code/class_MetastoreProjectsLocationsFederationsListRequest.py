from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetastoreProjectsLocationsFederationsListRequest(_messages.Message):
    """A MetastoreProjectsLocationsFederationsListRequest object.

  Fields:
    filter: Optional. The filter to apply to list results.
    orderBy: Optional. Specify the ordering of results as described in Sorting
      Order
      (https://cloud.google.com/apis/design/design_patterns#sorting_order). If
      not specified, the results will be sorted in the default order.
    pageSize: Optional. The maximum number of federations to return. The
      response may contain less than the maximum number. If unspecified, no
      more than 500 services are returned. The maximum value is 1000; values
      above 1000 are changed to 1000.
    pageToken: Optional. A page token, received from a previous
      ListFederationServices call. Provide this token to retrieve the
      subsequent page.To retrieve the first page, supply an empty page
      token.When paginating, other parameters provided to
      ListFederationServices must match the call that provided the page token.
    parent: Required. The relative resource name of the location of metastore
      federations to list, in the following form:
      projects/{project_number}/locations/{location_id}.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)