from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1ExplanationConfig(_messages.Message):
    """Message holding configuration options for explaining model predictions.
  There are three feature attribution methods supported for TensorFlow models:
  integrated gradients, sampled Shapley, and XRAI. [Learn more about feature
  attributions.](/ai-platform/prediction/docs/ai-explanations/overview)

  Fields:
    ablationAttribution: TensorFlow framework explanation methods. Deprecated.
      Attributes credit to model inputs by ablating features (ie. setting them
      to their default/missing values) and computing corresponding model score
      delta per feature. The term "ablation" is in reference to running an
      "ablation study" to analyze input effects on the outcome of interest,
      which in this case is the model's output. This attribution method is
      supported for TensorFlow and XGBoost models.
    integratedGradientsAttribution: Attributes credit by computing the Aumann-
      Shapley value taking advantage of the model's fully differentiable
      structure. Refer to this paper for more details:
      https://arxiv.org/abs/1703.01365
    saabasAttribution: Attributes credit by running a faster approximation to
      the TreeShap method. Please refer to this link for more details:
      https://blog.datadive.net/interpreting-random-forests/ This attribution
      method is only supported for XGBoost models.
    sampledShapleyAttribution: An attribution method that approximates Shapley
      values for features that contribute to the label being predicted. A
      sampling strategy is used to approximate the value rather than
      considering all subsets of features.
    treeShapAttribution: XGBoost framework explanation methods. Attributes
      credit by computing the Shapley value taking advantage of the model's
      tree ensemble structure. Refer to this paper for more details:
      http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-
      model-predictions.pdf. This attribution method is supported for XGBoost
      models.
    xraiAttribution: Attributes credit by computing the XRAI taking advantage
      of the model's fully differentiable structure. Refer to this paper for
      more details: https://arxiv.org/abs/1906.02825 Currently only
      implemented for models with natural image inputs.
  """
    ablationAttribution = _messages.MessageField('GoogleCloudMlV1AblationAttribution', 1)
    integratedGradientsAttribution = _messages.MessageField('GoogleCloudMlV1IntegratedGradientsAttribution', 2)
    saabasAttribution = _messages.MessageField('GoogleCloudMlV1SaabasAttribution', 3)
    sampledShapleyAttribution = _messages.MessageField('GoogleCloudMlV1SampledShapleyAttribution', 4)
    treeShapAttribution = _messages.MessageField('GoogleCloudMlV1TreeShapAttribution', 5)
    xraiAttribution = _messages.MessageField('GoogleCloudMlV1XraiAttribution', 6)