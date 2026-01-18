from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1EvaluateInstancesResponse(_messages.Message):
    """Response message for EvaluationService.EvaluateInstances.

  Fields:
    bleuResults: Results for bleu metric.
    coherenceResult: Result for coherence metric.
    exactMatchResults: Auto metric evaluation results. Results for exact match
      metric.
    fluencyResult: LLM-based metric evaluation result. General text generation
      metrics, applicable to other categories. Result for fluency metric.
    fulfillmentResult: Result for fulfillment metric.
    groundednessResult: Result for groundedness metric.
    pairwiseQuestionAnsweringQualityResult: Result for pairwise question
      answering quality metric.
    pairwiseSummarizationQualityResult: Result for pairwise summarization
      quality metric.
    questionAnsweringCorrectnessResult: Result for question answering
      correctness metric.
    questionAnsweringHelpfulnessResult: Result for question answering
      helpfulness metric.
    questionAnsweringQualityResult: Question answering only metrics. Result
      for question answering quality metric.
    questionAnsweringRelevanceResult: Result for question answering relevance
      metric.
    rougeResults: Results for rouge metric.
    safetyResult: Result for safety metric.
    summarizationHelpfulnessResult: Result for summarization helpfulness
      metric.
    summarizationQualityResult: Summarization only metrics. Result for
      summarization quality metric.
    summarizationVerbosityResult: Result for summarization verbosity metric.
    toolCallValidResults: Tool call metrics. Results for tool call valid
      metric.
    toolNameMatchResults: Results for tool name match metric.
    toolParameterKeyMatchResults: Results for tool parameter key match metric.
    toolParameterKvMatchResults: Results for tool parameter key value match
      metric.
  """
    bleuResults = _messages.MessageField('GoogleCloudAiplatformV1beta1BleuResults', 1)
    coherenceResult = _messages.MessageField('GoogleCloudAiplatformV1beta1CoherenceResult', 2)
    exactMatchResults = _messages.MessageField('GoogleCloudAiplatformV1beta1ExactMatchResults', 3)
    fluencyResult = _messages.MessageField('GoogleCloudAiplatformV1beta1FluencyResult', 4)
    fulfillmentResult = _messages.MessageField('GoogleCloudAiplatformV1beta1FulfillmentResult', 5)
    groundednessResult = _messages.MessageField('GoogleCloudAiplatformV1beta1GroundednessResult', 6)
    pairwiseQuestionAnsweringQualityResult = _messages.MessageField('GoogleCloudAiplatformV1beta1PairwiseQuestionAnsweringQualityResult', 7)
    pairwiseSummarizationQualityResult = _messages.MessageField('GoogleCloudAiplatformV1beta1PairwiseSummarizationQualityResult', 8)
    questionAnsweringCorrectnessResult = _messages.MessageField('GoogleCloudAiplatformV1beta1QuestionAnsweringCorrectnessResult', 9)
    questionAnsweringHelpfulnessResult = _messages.MessageField('GoogleCloudAiplatformV1beta1QuestionAnsweringHelpfulnessResult', 10)
    questionAnsweringQualityResult = _messages.MessageField('GoogleCloudAiplatformV1beta1QuestionAnsweringQualityResult', 11)
    questionAnsweringRelevanceResult = _messages.MessageField('GoogleCloudAiplatformV1beta1QuestionAnsweringRelevanceResult', 12)
    rougeResults = _messages.MessageField('GoogleCloudAiplatformV1beta1RougeResults', 13)
    safetyResult = _messages.MessageField('GoogleCloudAiplatformV1beta1SafetyResult', 14)
    summarizationHelpfulnessResult = _messages.MessageField('GoogleCloudAiplatformV1beta1SummarizationHelpfulnessResult', 15)
    summarizationQualityResult = _messages.MessageField('GoogleCloudAiplatformV1beta1SummarizationQualityResult', 16)
    summarizationVerbosityResult = _messages.MessageField('GoogleCloudAiplatformV1beta1SummarizationVerbosityResult', 17)
    toolCallValidResults = _messages.MessageField('GoogleCloudAiplatformV1beta1ToolCallValidResults', 18)
    toolNameMatchResults = _messages.MessageField('GoogleCloudAiplatformV1beta1ToolNameMatchResults', 19)
    toolParameterKeyMatchResults = _messages.MessageField('GoogleCloudAiplatformV1beta1ToolParameterKeyMatchResults', 20)
    toolParameterKvMatchResults = _messages.MessageField('GoogleCloudAiplatformV1beta1ToolParameterKVMatchResults', 21)