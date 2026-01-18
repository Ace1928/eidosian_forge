from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterOrganizationsSimulationsValuedResourcesListRequest(_messages.Message):
    """A SecuritycenterOrganizationsSimulationsValuedResourcesListRequest
  object.

  Fields:
    filter: The filter expression that filters the valued resources in the
      response. Supported fields: * `resource_value` supports = *
      `resource_type` supports =
    orderBy: Optional. The fields by which to order the valued resources
      response. Supported fields: * `exposed_score` * `resource_value` *
      `resource_type` Values should be a comma separated list of fields. For
      example: `exposed_score,resource_value`. The default sorting order is
      descending. To specify ascending or descending order for a field, append
      a " ASC" or a " DESC" suffix, respectively; for example: `exposed_score
      DESC`.
    pageSize: The maximum number of results to return in a single response.
      Default is 10, minimum is 1, maximum is 1000.
    pageToken: The value returned by the last `ListValuedResourcesResponse`;
      indicates that this is a continuation of a prior `ListValuedResources`
      call, and that the system should return the next page of data.
    parent: Required. Name of parent to list exposed resources. Valid formats:
      "organizations/{organization}",
      "organizations/{organization}/simulations/{simulation}" "organizations/{
      organization}/simulations/{simulation}/attackExposureResults/{attack_exp
      osure_result_v2}"
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)