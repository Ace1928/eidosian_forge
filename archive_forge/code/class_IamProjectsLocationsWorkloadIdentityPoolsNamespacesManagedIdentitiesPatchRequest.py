from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesPatchRequest(_messages.Message):
    """A IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesPa
  tchRequest object.

  Fields:
    name: Output only. The resource name of the managed identity.
    updateMask: Required. The list of fields to update.
    workloadIdentityPoolManagedIdentity: A WorkloadIdentityPoolManagedIdentity
      resource to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    updateMask = _messages.StringField(2)
    workloadIdentityPoolManagedIdentity = _messages.MessageField('WorkloadIdentityPoolManagedIdentity', 3)