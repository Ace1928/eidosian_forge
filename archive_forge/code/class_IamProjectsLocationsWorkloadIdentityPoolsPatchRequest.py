from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsLocationsWorkloadIdentityPoolsPatchRequest(_messages.Message):
    """A IamProjectsLocationsWorkloadIdentityPoolsPatchRequest object.

  Fields:
    googleIamV1betaWorkloadIdentityPool: A GoogleIamV1betaWorkloadIdentityPool
      resource to be passed as the request body.
    name: Output only. The resource name of the pool.
    updateMask: Required. The list of fields to update.
  """
    googleIamV1betaWorkloadIdentityPool = _messages.MessageField('GoogleIamV1betaWorkloadIdentityPool', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)