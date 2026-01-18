from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1Explanation(_messages.Message):
    """Explanation of a prediction (provided in PredictResponse.predictions)
  produced by the Model on a given instance.

  Fields:
    attributions: Output only. Feature attributions grouped by predicted
      outputs. For Models that predict only one output, such as regression
      Models that predict only one score, there is only one attibution that
      explains the predicted output. For Models that predict multiple outputs,
      such as multiclass Models that predict multiple classes, each element
      explains one specific item. Attribution.output_index can be used to
      identify which output this attribution is explaining. By default, we
      provide Shapley values for the predicted class. However, you can
      configure the explanation request to generate Shapley values for any
      other classes too. For example, if a model predicts a probability of
      `0.4` for approving a loan application, the model's decision is to
      reject the application since `p(reject) = 0.6 > p(approve) = 0.4`, and
      the default Shapley values would be computed for rejection decision and
      not approval, even though the latter might be the positive class. If
      users set ExplanationParameters.top_k, the attributions are sorted by
      instance_output_value in descending order. If
      ExplanationParameters.output_indices is specified, the attributions are
      stored by Attribution.output_index in the same order as they appear in
      the output_indices.
    neighbors: Output only. List of the nearest neighbors for example-based
      explanations. For models deployed with the examples explanations feature
      enabled, the attributions field is empty and instead the neighbors field
      is populated.
  """
    attributions = _messages.MessageField('GoogleCloudAiplatformV1beta1Attribution', 1, repeated=True)
    neighbors = _messages.MessageField('GoogleCloudAiplatformV1beta1Neighbor', 2, repeated=True)