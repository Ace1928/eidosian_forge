from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalClusterUpgradePolicy(_messages.Message):
    """BareMetalClusterUpgradePolicy defines the cluster upgrade policy.

  Enums:
    PolicyValueValuesEnum: Specifies which upgrade policy to use.

  Fields:
    controlPlaneOnly: Controls whether upgrade applies to only the control
      plane.
    pause: Output only. Pause is used to show the upgrade pause status. It's
      view only for now.
    policy: Specifies which upgrade policy to use.
  """

    class PolicyValueValuesEnum(_messages.Enum):
        """Specifies which upgrade policy to use.

    Values:
      NODE_POOL_POLICY_UNSPECIFIED: No upgrade policy selected.
      SERIAL: Upgrade worker node pools sequentially.
      CONCURRENT: Upgrade all worker node pools in parallel.
    """
        NODE_POOL_POLICY_UNSPECIFIED = 0
        SERIAL = 1
        CONCURRENT = 2
    controlPlaneOnly = _messages.BooleanField(1)
    pause = _messages.BooleanField(2)
    policy = _messages.EnumField('PolicyValueValuesEnum', 3)