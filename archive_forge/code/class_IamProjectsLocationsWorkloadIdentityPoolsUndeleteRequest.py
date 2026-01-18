from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsLocationsWorkloadIdentityPoolsUndeleteRequest(_messages.Message):
    """A IamProjectsLocationsWorkloadIdentityPoolsUndeleteRequest object.

  Fields:
    googleIamV1betaUndeleteWorkloadIdentityPoolRequest: A
      GoogleIamV1betaUndeleteWorkloadIdentityPoolRequest resource to be passed
      as the request body.
    name: Required. The name of the pool to undelete.
  """
    googleIamV1betaUndeleteWorkloadIdentityPoolRequest = _messages.MessageField('GoogleIamV1betaUndeleteWorkloadIdentityPoolRequest', 1)
    name = _messages.StringField(2, required=True)