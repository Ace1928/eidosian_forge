from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1Examples(_messages.Message):
    """Example-based explainability that returns the nearest neighbors from the
  provided dataset.

  Fields:
    exampleGcsSource: The Cloud Storage input instances.
    nearestNeighborSearchConfig: The full configuration for the generated
      index, the semantics are the same as metadata and should match
      [NearestNeighborSearchConfig](https://cloud.google.com/vertex-
      ai/docs/explainable-ai/configuring-explanations-example-based#nearest-
      neighbor-search-config).
    neighborCount: The number of neighbors to return when querying for
      examples.
    presets: Simplified preset configuration, which automatically sets
      configuration values based on the desired query speed-precision trade-
      off and modality.
  """
    exampleGcsSource = _messages.MessageField('GoogleCloudAiplatformV1ExamplesExampleGcsSource', 1)
    nearestNeighborSearchConfig = _messages.MessageField('extra_types.JsonValue', 2)
    neighborCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    presets = _messages.MessageField('GoogleCloudAiplatformV1Presets', 4)