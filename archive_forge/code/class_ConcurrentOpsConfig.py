from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConcurrentOpsConfig(_messages.Message):
    """Configures operations using concurrent ops.

  Fields:
    enableConcurrentCreateNodePool: Enables concurrent ops for supported
      CreateNodePool cases. Some fields may still use legacy ops.
    enableConcurrentDeleteNodePool: Enables concurrent ops for supported
      DeleteNodePool cases. Some fields may still use legacy ops.
    enableConcurrentResizeNodePool: Enables concurrent ops for ResizeNodePool
      operations.
    enableConcurrentRollbackNodePool: Enables concurrent ops for supported
      RollbackNodePool cases. Some fields may still use legacy ops.
    enableConcurrentUpdateNodePoolVersion: Enables concurrent ops for
      UpdateNodePool with only the version field. Some cluster features may
      still use legacy ops.
  """
    enableConcurrentCreateNodePool = _messages.BooleanField(1)
    enableConcurrentDeleteNodePool = _messages.BooleanField(2)
    enableConcurrentResizeNodePool = _messages.BooleanField(3)
    enableConcurrentRollbackNodePool = _messages.BooleanField(4)
    enableConcurrentUpdateNodePoolVersion = _messages.BooleanField(5)