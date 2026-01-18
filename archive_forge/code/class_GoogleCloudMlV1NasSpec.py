from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1NasSpec(_messages.Message):
    """Spec for Neural Architecture Search (NAS) jobs.

  Fields:
    multiTrialAlgorithmSpec: The spec of multi-trial algorithms.
    oneShotAlgorithmSpec: The spec of one-shot algorithms.
    previousNasJobId: The previous NAS job ID to resume search. The
      `search_space_spec` needs to be the same between this and previous NAS
      job and its job state is `FINISHED` or `CANCELLED`.
    searchSpaceSpec: Required. It defines the search space for Neural
      Architecture Search (NAS).
  """
    multiTrialAlgorithmSpec = _messages.MessageField('GoogleCloudMlV1NasSpecMultiTrialAlgorithmSpec', 1)
    oneShotAlgorithmSpec = _messages.MessageField('GoogleCloudMlV1NasSpecOneShotAlgorithmSpec', 2)
    previousNasJobId = _messages.StringField(3)
    searchSpaceSpec = _messages.StringField(4)