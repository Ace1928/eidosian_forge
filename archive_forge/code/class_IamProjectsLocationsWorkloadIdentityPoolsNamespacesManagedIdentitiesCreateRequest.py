from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesCreateRequest(_messages.Message):
    """A IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesCr
  eateRequest object.

  Fields:
    parent: Required. The parent resource to create the manage identity in.
      The only supported location is `global`.
    workloadIdentityPoolManagedIdentity: A WorkloadIdentityPoolManagedIdentity
      resource to be passed as the request body.
    workloadIdentityPoolManagedIdentityId: Required. The ID to use for the
      managed identity. This value must: * contain at most 63 characters *
      contain only lowercase alphanumeric characters or `-` * start with an
      alphanumeric character * end with an alphanumeric character The prefix
      "gcp-" will be reserved for future uses.
  """
    parent = _messages.StringField(1, required=True)
    workloadIdentityPoolManagedIdentity = _messages.MessageField('WorkloadIdentityPoolManagedIdentity', 2)
    workloadIdentityPoolManagedIdentityId = _messages.StringField(3)