from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EthereumEndpoints(_messages.Message):
    """Contains endpoint information specific to Ethereum nodes.

  Fields:
    beaconApiEndpoint: Output only. The assigned URL for the node's Beacon API
      endpoint.
    beaconPrometheusMetricsApiEndpoint: Output only. The assigned URL for the
      node's Beacon Prometheus metrics endpoint. See [Prometheus
      Metrics](https://lighthouse-book.sigmaprime.io/advanced_metrics.html)
      for more details.
    executionClientPrometheusMetricsApiEndpoint: Output only. The assigned URL
      for the node's execution client's Prometheus metrics endpoint.
  """
    beaconApiEndpoint = _messages.StringField(1)
    beaconPrometheusMetricsApiEndpoint = _messages.StringField(2)
    executionClientPrometheusMetricsApiEndpoint = _messages.StringField(3)