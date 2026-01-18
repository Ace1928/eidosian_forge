from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2EvaluationConfigSmartComposeConfig(_messages.Message):
    """Smart compose specific configuration for evaluation job.

  Fields:
    allowlistDocument: The allowlist document resource name. Format:
      `projects//knowledgeBases//documents/`. Only used for smart compose
      model.
    maxResultCount: Required. The model to be evaluated can return multiple
      results with confidence score on each query. These results will be
      sorted by the descending order of the scores and we only keep the first
      max_result_count results as the final results to evaluate.
  """
    allowlistDocument = _messages.StringField(1)
    maxResultCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)