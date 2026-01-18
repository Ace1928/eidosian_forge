from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1PublisherModelCallToActionOpenFineTuningPipelines(_messages.Message):
    """Open fine tuning pipelines.

  Fields:
    fineTuningPipelines: Required. Regional resource references to fine tuning
      pipelines.
  """
    fineTuningPipelines = _messages.MessageField('GoogleCloudAiplatformV1beta1PublisherModelCallToActionRegionalResourceReferences', 1, repeated=True)