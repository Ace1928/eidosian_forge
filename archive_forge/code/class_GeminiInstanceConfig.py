from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GeminiInstanceConfig(_messages.Message):
    """Gemini instance configuration.

  Fields:
    activeQueryEnabled: Output only. Whether the active query is enabled.
    entitled: Output only. Whether Gemini is enabled.
    flagRecommenderEnabled: Output only. Whether the flag recommender is
      enabled.
    googleVacuumMgmtEnabled: Output only. Whether the vacuum management is
      enabled.
    indexAdvisorEnabled: Output only. Whether the index advisor is enabled.
    oomSessionCancelEnabled: Output only. Whether canceling the out-of-memory
      (OOM) session is enabled.
  """
    activeQueryEnabled = _messages.BooleanField(1)
    entitled = _messages.BooleanField(2)
    flagRecommenderEnabled = _messages.BooleanField(3)
    googleVacuumMgmtEnabled = _messages.BooleanField(4)
    indexAdvisorEnabled = _messages.BooleanField(5)
    oomSessionCancelEnabled = _messages.BooleanField(6)