from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1MonitoringConfig(_messages.Message):
    """Parameters that describe the Monitoring configuration in a cluster.

  Fields:
    managedPrometheusConfig: Enable Google Cloud Managed Service for
      Prometheus in the cluster.
  """
    managedPrometheusConfig = _messages.MessageField('GoogleCloudGkemulticloudV1ManagedPrometheusConfig', 1)