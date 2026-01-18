from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Canary(_messages.Message):
    """Canary represents the canary deployment strategy.

  Fields:
    canaryDeployment: Configures the progressive based deployment for a
      Target.
    customCanaryDeployment: Configures the progressive based deployment for a
      Target, but allows customizing at the phase level where a phase
      represents each of the percentage deployments.
    runtimeConfig: Optional. Runtime specific configurations for the
      deployment strategy. The runtime configuration is used to determine how
      Cloud Deploy will split traffic to enable a progressive deployment.
  """
    canaryDeployment = _messages.MessageField('CanaryDeployment', 1)
    customCanaryDeployment = _messages.MessageField('CustomCanaryDeployment', 2)
    runtimeConfig = _messages.MessageField('RuntimeConfig', 3)