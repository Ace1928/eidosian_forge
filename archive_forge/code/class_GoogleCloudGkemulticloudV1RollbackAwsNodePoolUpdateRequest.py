from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1RollbackAwsNodePoolUpdateRequest(_messages.Message):
    """Request message for `AwsClusters.RollbackAwsNodePoolUpdate` method.

  Fields:
    respectPdb: Optional. Option for rollback to ignore the
      PodDisruptionBudget when draining the node pool nodes. Default value is
      false.
  """
    respectPdb = _messages.BooleanField(1)