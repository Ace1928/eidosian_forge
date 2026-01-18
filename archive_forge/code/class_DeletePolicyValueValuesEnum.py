from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeletePolicyValueValuesEnum(_messages.Enum):
    """Optional. Policy on how resources actuated by the deployment should be
    deleted. If unspecified, the default behavior is to delete the underlying
    resources.

    Values:
      DELETE_POLICY_UNSPECIFIED: Unspecified policy, resources will be
        deleted.
      DELETE: Deletes resources actuated by the deployment.
      ABANDON: Abandons resources and only deletes the deployment and its
        metadata.
    """
    DELETE_POLICY_UNSPECIFIED = 0
    DELETE = 1
    ABANDON = 2