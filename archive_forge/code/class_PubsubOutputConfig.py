from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PubsubOutputConfig(_messages.Message):
    """Configuration for the output that is specific to Pub/Sub when choosing a
  Pub/Sub queue as the output destination.

  Fields:
    openTelemetryFormat: open_telemetry_format contains additional information
      needed to convert Cloud Trace data.
  """
    openTelemetryFormat = _messages.MessageField('OpenTelemetryFormat', 1)