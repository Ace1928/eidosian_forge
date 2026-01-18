from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CommonFeatureState(_messages.Message):
    """CommonFeatureState contains Hub-wide Feature status information.

  Fields:
    appdevexperience: Appdevexperience specific state.
    clusterupgrade: ClusterUpgrade fleet-level state.
    fleetobservability: FleetObservability feature state.
    helloworld: Hello World-specific state.
    state: Output only. The "running state" of the Feature in this Hub.
  """
    appdevexperience = _messages.MessageField('AppDevExperienceFeatureState', 1)
    clusterupgrade = _messages.MessageField('ClusterUpgradeFleetState', 2)
    fleetobservability = _messages.MessageField('FleetObservabilityFeatureState', 3)
    helloworld = _messages.MessageField('HelloWorldFeatureState', 4)
    state = _messages.MessageField('FeatureState', 5)