from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2EvaluationConfig(_messages.Message):
    """The configuration for model evaluation.

  Fields:
    datasets: Required. Datasets used for evaluation.
    smartComposeConfig: Configuration for smart compose model evalution.
    smartReplyConfig: Configuration for smart reply model evalution.
  """
    datasets = _messages.MessageField('GoogleCloudDialogflowV2InputDataset', 1, repeated=True)
    smartComposeConfig = _messages.MessageField('GoogleCloudDialogflowV2EvaluationConfigSmartComposeConfig', 2)
    smartReplyConfig = _messages.MessageField('GoogleCloudDialogflowV2EvaluationConfigSmartReplyConfig', 3)