from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesWorkloadSourcesDeleteRequest(_messages.Message):
    """A IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesWo
  rkloadSourcesDeleteRequest object.

  Fields:
    etag: Optional. The etag for this workload source. If provided, it must
      match the server's etag.
    name: Required. The name of the workload source to delete.
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2, required=True)