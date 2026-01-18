from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrivatePoolV1Config(_messages.Message):
    """Configuration for a V1 `PrivatePool`.

  Fields:
    networkConfig: Network configuration for the pool.
    privateServiceConnect: Immutable. Private Service Connect(PSC) Network
      configuration for the pool.
    workerConfig: Machine configuration for the workers in the pool.
  """
    networkConfig = _messages.MessageField('NetworkConfig', 1)
    privateServiceConnect = _messages.MessageField('PrivateServiceConnect', 2)
    workerConfig = _messages.MessageField('WorkerConfig', 3)