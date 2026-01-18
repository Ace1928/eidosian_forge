from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FleetObservabilityFleetObservabilityMonitoringState(_messages.Message):
    """Feature state for monitoring feature.

  Fields:
    state: The base feature state of fleet monitoring feature.
  """
    state = _messages.MessageField('FleetObservabilityFleetObservabilityBaseFeatureState', 1)