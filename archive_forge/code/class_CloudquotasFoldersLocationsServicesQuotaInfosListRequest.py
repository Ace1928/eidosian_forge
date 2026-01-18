from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudquotasFoldersLocationsServicesQuotaInfosListRequest(_messages.Message):
    """A CloudquotasFoldersLocationsServicesQuotaInfosListRequest object.

  Fields:
    pageSize: Optional. Requested page size. Server may return fewer items
      than requested. If unspecified, server will pick an appropriate default.
    pageToken: Optional. A token identifying a page of results the server
      should return.
    parent: Required. Parent value of QuotaInfo resources. Listing across
      different resource containers (such as 'projects/-') is not allowed.
      Example names:
      `projects/123/locations/global/services/compute.googleapis.com`
      `folders/234/locations/global/services/compute.googleapis.com`
      `organizations/345/locations/global/services/compute.googleapis.com`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)