from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FleetObservabilityFeatureSpec(_messages.Message):
    """**Fleet Observability**: The Hub-wide input for the FleetObservability
  feature.

  Fields:
    loggingConfig: Specified if fleet logging feature is enabled for the
      entire fleet. If UNSPECIFIED, fleet logging feature is disabled for the
      entire fleet.
  """
    loggingConfig = _messages.MessageField('FleetObservabilityLoggingConfig', 1)