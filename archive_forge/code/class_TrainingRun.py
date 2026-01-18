from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TrainingRun(_messages.Message):
    """Information about a single training query run for the model.

  Fields:
    classLevelGlobalExplanations: Output only. Global explanation contains the
      explanation of top features on the class level. Applies to
      classification models only.
    dataSplitResult: Output only. Data split result of the training run. Only
      set when the input data is actually split.
    evaluationMetrics: Output only. The evaluation metrics over training/eval
      data that were computed at the end of training.
    modelLevelGlobalExplanation: Output only. Global explanation contains the
      explanation of top features on the model level. Applies to both
      regression and classification models.
    results: Output only. Output of each iteration run, results.size() <=
      max_iterations.
    startTime: Output only. The start time of this training run.
    trainingOptions: Output only. Options that were used for this training
      run, includes user specified and default options that were used.
    trainingStartTime: Output only. The start time of this training run, in
      milliseconds since epoch.
    vertexAiModelId: The model id in the [Vertex AI Model
      Registry](https://cloud.google.com/vertex-ai/docs/model-
      registry/introduction) for this training run.
    vertexAiModelVersion: Output only. The model version in the [Vertex AI
      Model Registry](https://cloud.google.com/vertex-ai/docs/model-
      registry/introduction) for this training run.
  """
    classLevelGlobalExplanations = _messages.MessageField('GlobalExplanation', 1, repeated=True)
    dataSplitResult = _messages.MessageField('DataSplitResult', 2)
    evaluationMetrics = _messages.MessageField('EvaluationMetrics', 3)
    modelLevelGlobalExplanation = _messages.MessageField('GlobalExplanation', 4)
    results = _messages.MessageField('IterationResult', 5, repeated=True)
    startTime = _messages.StringField(6)
    trainingOptions = _messages.MessageField('TrainingOptions', 7)
    trainingStartTime = _messages.IntegerField(8)
    vertexAiModelId = _messages.StringField(9)
    vertexAiModelVersion = _messages.StringField(10)