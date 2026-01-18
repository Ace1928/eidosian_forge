from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsSecurityincidentenvironmentsListRequest(_messages.Message):
    """A ApigeeOrganizationsSecurityincidentenvironmentsListRequest object.

  Fields:
    filter: Optional. Filter list security incident stats per environment by
      time range "first_detected_time >", "last_detected_time <"
    orderBy: Optional. Field to sort by. See
      https://google.aip.dev/132#ordering for more details. If not specified,
      the results will be sorted in the default order.
    pageSize: Optional. The maximum number of environments to return. The
      service may return fewer than this value. If unspecified, at most 50
      environments will be returned.
    pageToken: Optional. A page token, received from a previous
      `ListSecurityIncidentEnvironments` call. Provide this to retrieve the
      subsequent page.
    parent: Required. For a specific organization, list all environments with
      security incidents stats. Format: `organizations/{org}}`
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)