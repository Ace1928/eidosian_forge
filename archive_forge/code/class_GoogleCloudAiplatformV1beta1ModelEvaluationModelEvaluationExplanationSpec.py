from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ModelEvaluationModelEvaluationExplanationSpec(_messages.Message):
    """A
  GoogleCloudAiplatformV1beta1ModelEvaluationModelEvaluationExplanationSpec
  object.

  Fields:
    explanationSpec: Explanation spec details.
    explanationType: Explanation type. For AutoML Image Classification models,
      possible values are: * `image-integrated-gradients` * `image-xrai`
  """
    explanationSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1ExplanationSpec', 1)
    explanationType = _messages.StringField(2)