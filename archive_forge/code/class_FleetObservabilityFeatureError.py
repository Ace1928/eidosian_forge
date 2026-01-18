from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FleetObservabilityFeatureError(_messages.Message):
    """All error details of the fleet observability feature.

  Fields:
    code: The code of the error.
    description: A human-readable description of the current status.
  """
    code = _messages.StringField(1)
    description = _messages.StringField(2)