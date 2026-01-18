from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1ListOptimalTrialsResponse(_messages.Message):
    """The response message for the ListOptimalTrials method.

  Fields:
    trials: The pareto-optimal trials for multiple objective study or the
      optimal trial for single objective study. The definition of pareto-
      optimal can be checked in wiki page.
      https://en.wikipedia.org/wiki/Pareto_efficiency
  """
    trials = _messages.MessageField('GoogleCloudMlV1Trial', 1, repeated=True)