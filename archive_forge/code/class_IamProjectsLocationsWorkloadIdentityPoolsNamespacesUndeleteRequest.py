from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsLocationsWorkloadIdentityPoolsNamespacesUndeleteRequest(_messages.Message):
    """A IamProjectsLocationsWorkloadIdentityPoolsNamespacesUndeleteRequest
  object.

  Fields:
    name: Required. The name of the namespace to undelete.
    undeleteWorkloadIdentityPoolNamespaceRequest: A
      UndeleteWorkloadIdentityPoolNamespaceRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    undeleteWorkloadIdentityPoolNamespaceRequest = _messages.MessageField('UndeleteWorkloadIdentityPoolNamespaceRequest', 2)