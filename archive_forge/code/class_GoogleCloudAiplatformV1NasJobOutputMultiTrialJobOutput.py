from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1NasJobOutputMultiTrialJobOutput(_messages.Message):
    """The output of a multi-trial Neural Architecture Search (NAS) jobs.

  Fields:
    searchTrials: Output only. List of NasTrials that were started as part of
      search stage.
    trainTrials: Output only. List of NasTrials that were started as part of
      train stage.
  """
    searchTrials = _messages.MessageField('GoogleCloudAiplatformV1NasTrial', 1, repeated=True)
    trainTrials = _messages.MessageField('GoogleCloudAiplatformV1NasTrial', 2, repeated=True)