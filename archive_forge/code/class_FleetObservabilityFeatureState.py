from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FleetObservabilityFeatureState(_messages.Message):
    """**FleetObservability**: Hub-wide Feature for FleetObservability feature.
  state.

  Fields:
    logging: The feature state of default logging.
    monitoring: The feature state of fleet monitoring.
  """
    logging = _messages.MessageField('FleetObservabilityFleetObservabilityLoggingState', 1)
    monitoring = _messages.MessageField('FleetObservabilityFleetObservabilityMonitoringState', 2)