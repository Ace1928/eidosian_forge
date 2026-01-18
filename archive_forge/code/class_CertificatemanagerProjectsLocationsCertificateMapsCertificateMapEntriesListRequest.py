from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificatemanagerProjectsLocationsCertificateMapsCertificateMapEntriesListRequest(_messages.Message):
    """A CertificatemanagerProjectsLocationsCertificateMapsCertificateMapEntrie
  sListRequest object.

  Fields:
    filter: Filter expression to restrict the returned Certificate Map
      Entries.
    orderBy: A list of Certificate Map Entry field names used to specify the
      order of the returned results. The default sorting order is ascending.
      To specify descending order for a field, add a suffix " desc".
    pageSize: Maximum number of certificate map entries to return. The service
      may return fewer than this value. If unspecified, at most 50 certificate
      map entries will be returned. The maximum value is 1000; values above
      1000 will be coerced to 1000.
    pageToken: The value returned by the last
      `ListCertificateMapEntriesResponse`. Indicates that this is a
      continuation of a prior `ListCertificateMapEntries` call, and that the
      system should return the next page of data.
    parent: Required. The project, location and certificate map from which the
      certificate map entries should be listed, specified in the format
      `projects/*/locations/*/certificateMaps/*`.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)