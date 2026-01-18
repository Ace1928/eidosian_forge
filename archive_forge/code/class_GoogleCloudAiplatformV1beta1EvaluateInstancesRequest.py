from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1EvaluateInstancesRequest(_messages.Message):
    """Request message for EvaluationService.EvaluateInstances.

  Fields:
    bleuInput: Instances and metric spec for bleu metric.
    coherenceInput: Input for coherence metric.
    exactMatchInput: Auto metric instances. Instances and metric spec for
      exact match metric.
    fluencyInput: LLM-based metric instance. General text generation metrics,
      applicable to other categories. Input for fluency metric.
    fulfillmentInput: Input for fulfillment metric.
    groundednessInput: Input for groundedness metric.
    pairwiseQuestionAnsweringQualityInput: Input for pairwise question
      answering quality metric.
    pairwiseSummarizationQualityInput: Input for pairwise summarization
      quality metric.
    questionAnsweringCorrectnessInput: Input for question answering
      correctness metric.
    questionAnsweringHelpfulnessInput: Input for question answering
      helpfulness metric.
    questionAnsweringQualityInput: Input for question answering quality
      metric.
    questionAnsweringRelevanceInput: Input for question answering relevance
      metric.
    rougeInput: Instances and metric spec for rouge metric.
    safetyInput: Input for safety metric.
    summarizationHelpfulnessInput: Input for summarization helpfulness metric.
    summarizationQualityInput: Input for summarization quality metric.
    summarizationVerbosityInput: Input for summarization verbosity metric.
    toolCallValidInput: Tool call metric instances. Input for tool call valid
      metric.
    toolNameMatchInput: Input for tool name match metric.
    toolParameterKeyMatchInput: Input for tool parameter key match metric.
    toolParameterKvMatchInput: Input for tool parameter key value match
      metric.
  """
    bleuInput = _messages.MessageField('GoogleCloudAiplatformV1beta1BleuInput', 1)
    coherenceInput = _messages.MessageField('GoogleCloudAiplatformV1beta1CoherenceInput', 2)
    exactMatchInput = _messages.MessageField('GoogleCloudAiplatformV1beta1ExactMatchInput', 3)
    fluencyInput = _messages.MessageField('GoogleCloudAiplatformV1beta1FluencyInput', 4)
    fulfillmentInput = _messages.MessageField('GoogleCloudAiplatformV1beta1FulfillmentInput', 5)
    groundednessInput = _messages.MessageField('GoogleCloudAiplatformV1beta1GroundednessInput', 6)
    pairwiseQuestionAnsweringQualityInput = _messages.MessageField('GoogleCloudAiplatformV1beta1PairwiseQuestionAnsweringQualityInput', 7)
    pairwiseSummarizationQualityInput = _messages.MessageField('GoogleCloudAiplatformV1beta1PairwiseSummarizationQualityInput', 8)
    questionAnsweringCorrectnessInput = _messages.MessageField('GoogleCloudAiplatformV1beta1QuestionAnsweringCorrectnessInput', 9)
    questionAnsweringHelpfulnessInput = _messages.MessageField('GoogleCloudAiplatformV1beta1QuestionAnsweringHelpfulnessInput', 10)
    questionAnsweringQualityInput = _messages.MessageField('GoogleCloudAiplatformV1beta1QuestionAnsweringQualityInput', 11)
    questionAnsweringRelevanceInput = _messages.MessageField('GoogleCloudAiplatformV1beta1QuestionAnsweringRelevanceInput', 12)
    rougeInput = _messages.MessageField('GoogleCloudAiplatformV1beta1RougeInput', 13)
    safetyInput = _messages.MessageField('GoogleCloudAiplatformV1beta1SafetyInput', 14)
    summarizationHelpfulnessInput = _messages.MessageField('GoogleCloudAiplatformV1beta1SummarizationHelpfulnessInput', 15)
    summarizationQualityInput = _messages.MessageField('GoogleCloudAiplatformV1beta1SummarizationQualityInput', 16)
    summarizationVerbosityInput = _messages.MessageField('GoogleCloudAiplatformV1beta1SummarizationVerbosityInput', 17)
    toolCallValidInput = _messages.MessageField('GoogleCloudAiplatformV1beta1ToolCallValidInput', 18)
    toolNameMatchInput = _messages.MessageField('GoogleCloudAiplatformV1beta1ToolNameMatchInput', 19)
    toolParameterKeyMatchInput = _messages.MessageField('GoogleCloudAiplatformV1beta1ToolParameterKeyMatchInput', 20)
    toolParameterKvMatchInput = _messages.MessageField('GoogleCloudAiplatformV1beta1ToolParameterKVMatchInput', 21)