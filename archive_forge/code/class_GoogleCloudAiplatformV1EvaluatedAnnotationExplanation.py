from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1EvaluatedAnnotationExplanation(_messages.Message):
    """Explanation result of the prediction produced by the Model.

  Fields:
    explanation: Explanation attribution response details.
    explanationType: Explanation type. For AutoML Image Classification models,
      possible values are: * `image-integrated-gradients` * `image-xrai`
  """
    explanation = _messages.MessageField('GoogleCloudAiplatformV1Explanation', 1)
    explanationType = _messages.StringField(2)