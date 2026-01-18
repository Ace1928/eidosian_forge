from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceUsageExportConfig(_messages.Message):
    """Configuration for exporting cluster resource usages.

  Fields:
    bigqueryDestination: Configuration to use BigQuery as usage export
      destination.
    consumptionMeteringConfig: Configuration to enable resource consumption
      metering.
    enableNetworkEgressMetering: Whether to enable network egress metering for
      this cluster. If enabled, a daemonset will be created in the cluster to
      meter network egress traffic.
  """
    bigqueryDestination = _messages.MessageField('BigQueryDestination', 1)
    consumptionMeteringConfig = _messages.MessageField('ConsumptionMeteringConfig', 2)
    enableNetworkEgressMetering = _messages.BooleanField(3)