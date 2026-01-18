from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrivatecaProjectsLocationsCaPoolsCertificatesListRequest(_messages.Message):
    """A PrivatecaProjectsLocationsCaPoolsCertificatesListRequest object.

  Fields:
    filter: Optional. Only include resources that match the filter in the
      response. For details on supported filters and syntax, see [Certificates
      Filtering documentation](https://cloud.google.com/certificate-authority-
      service/docs/sorting-filtering-certificates#filtering_support).
    orderBy: Optional. Specify how the results should be sorted. For details
      on supported fields and syntax, see [Certificates Sorting
      documentation](https://cloud.google.com/certificate-authority-
      service/docs/sorting-filtering-certificates#sorting_support).
    pageSize: Optional. Limit on the number of Certificates to include in the
      response. Further Certificates can subsequently be obtained by including
      the ListCertificatesResponse.next_page_token in a subsequent request. If
      unspecified, the server will pick an appropriate default.
    pageToken: Optional. Pagination token, returned earlier via
      ListCertificatesResponse.next_page_token.
    parent: Required. The resource name of the location associated with the
      Certificates, in the format `projects/*/locations/*/caPools/*`.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)