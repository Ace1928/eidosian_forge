from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsLocationsWorkloadIdentityPoolsNamespacesPatchRequest(_messages.Message):
    """A IamProjectsLocationsWorkloadIdentityPoolsNamespacesPatchRequest
  object.

  Fields:
    name: Output only. The resource name of the namespace.
    updateMask: Required. The list of fields to update.
    workloadIdentityPoolNamespace: A WorkloadIdentityPoolNamespace resource to
      be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    updateMask = _messages.StringField(2)
    workloadIdentityPoolNamespace = _messages.MessageField('WorkloadIdentityPoolNamespace', 3)