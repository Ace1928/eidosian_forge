from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusteringMetrics(_messages.Message):
    """Evaluation metrics for clustering models.

  Fields:
    clusters: Information for all clusters.
    daviesBouldinIndex: Davies-Bouldin index.
    meanSquaredDistance: Mean of squared distances between each sample to its
      cluster centroid.
  """
    clusters = _messages.MessageField('Cluster', 1, repeated=True)
    daviesBouldinIndex = _messages.FloatField(2)
    meanSquaredDistance = _messages.FloatField(3)