from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1ManagedPrometheusConfig(_messages.Message):
    """ManagedPrometheusConfig defines the configuration for Google Cloud
  Managed Service for Prometheus.

  Fields:
    enabled: Enable Managed Collection.
  """
    enabled = _messages.BooleanField(1)