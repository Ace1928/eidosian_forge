from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConsumptionMeteringConfig(_messages.Message):
    """Parameters for controlling consumption metering.

  Fields:
    enabled: Whether to enable consumption metering for this cluster. If
      enabled, a second BigQuery table will be created to hold resource
      consumption records.
  """
    enabled = _messages.BooleanField(1)