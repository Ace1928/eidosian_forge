from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalNodePoolUpgradePolicy(_messages.Message):
    """BareMetalNodePoolUpgradePolicy defines the node pool upgrade policy.

  Fields:
    independent: Specify the intent to upgrade the node pool with or without
      the control plane upgrade. Defaults to false i.e. upgrade the node pool
      with control plane upgrade. Set this to true to upgrade or downgrade the
      node pool independently from the control plane.
    parallelUpgradeConfig: The parallel upgrade settings for worker node
      pools.
  """
    independent = _messages.BooleanField(1)
    parallelUpgradeConfig = _messages.MessageField('BareMetalParallelUpgradeConfig', 2)