from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedPrometheusConfig(_messages.Message):
    """ManagedPrometheusConfig defines the configuration for Google Cloud
  Managed Service for Prometheus.

  Fields:
    autoMonitoringConfig: GKE Workload Auto-Monitoring Configuration.
    enabled: Enable Managed Collection.
  """
    autoMonitoringConfig = _messages.MessageField('AutoMonitoringConfig', 1)
    enabled = _messages.BooleanField(2)