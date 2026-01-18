from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AssuredworkloadsOrganizationsLocationsWorkloadsDeleteRequest(_messages.Message):
    """A AssuredworkloadsOrganizationsLocationsWorkloadsDeleteRequest object.

  Fields:
    etag: Optional. The etag of the workload. If this is provided, it must
      match the server's etag.
    name: Required. The `name` field is used to identify the workload. Format:
      organizations/{org_id}/locations/{location_id}/workloads/{workload_id}
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2, required=True)