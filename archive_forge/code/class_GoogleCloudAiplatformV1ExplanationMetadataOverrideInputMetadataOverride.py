from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ExplanationMetadataOverrideInputMetadataOverride(_messages.Message):
    """The input metadata entries to be overridden.

  Fields:
    inputBaselines: Baseline inputs for this feature. This overrides the
      `input_baseline` field of the ExplanationMetadata.InputMetadata object
      of the corresponding feature's input metadata. If it's not specified,
      the original baselines are not overridden.
  """
    inputBaselines = _messages.MessageField('extra_types.JsonValue', 1, repeated=True)