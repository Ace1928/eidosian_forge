from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsLocationsWorkloadIdentityPoolsProvidersCreateRequest(_messages.Message):
    """A IamProjectsLocationsWorkloadIdentityPoolsProvidersCreateRequest
  object.

  Fields:
    googleIamV1betaWorkloadIdentityPoolProvider: A
      GoogleIamV1betaWorkloadIdentityPoolProvider resource to be passed as the
      request body.
    parent: Required. The pool to create this provider in.
    workloadIdentityPoolProviderId: Required. The ID for the provider, which
      becomes the final component of the resource name. This value must be
      4-32 characters, and may contain the characters [a-z0-9-]. The prefix
      `gcp-` is reserved for use by Google, and may not be specified.
  """
    googleIamV1betaWorkloadIdentityPoolProvider = _messages.MessageField('GoogleIamV1betaWorkloadIdentityPoolProvider', 1)
    parent = _messages.StringField(2, required=True)
    workloadIdentityPoolProviderId = _messages.StringField(3)