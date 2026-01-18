from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1NasJobSpecMultiTrialAlgorithmSpecSearchTrialSpec(_messages.Message):
    """Represent spec for search trials.

  Fields:
    maxFailedTrialCount: The number of failed trials that need to be seen
      before failing the NasJob. If set to 0, Vertex AI decides how many
      trials must fail before the whole job fails.
    maxParallelTrialCount: Required. The maximum number of trials to run in
      parallel.
    maxTrialCount: Required. The maximum number of Neural Architecture Search
      (NAS) trials to run.
    searchTrialJobSpec: Required. The spec of a search trial job. The same
      spec applies to all search trials.
  """
    maxFailedTrialCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    maxParallelTrialCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    maxTrialCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    searchTrialJobSpec = _messages.MessageField('GoogleCloudAiplatformV1CustomJobSpec', 4)