from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1UpdateExplanationDatasetRequest(_messages.Message):
    """Request message for ModelService.UpdateExplanationDataset.

  Fields:
    examples: The example config containing the location of the dataset.
  """
    examples = _messages.MessageField('GoogleCloudAiplatformV1beta1Examples', 1)