from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ModelExtractOptions(_messages.Message):
    """Options related to model extraction.

  Fields:
    trialId: The 1-based ID of the trial to be exported from a hyperparameter
      tuning model. If not specified, the trial with id = [Model](/bigquery/do
      cs/reference/rest/v2/models#resource:-model).defaultTrialId is exported.
      This field is ignored for models not trained with hyperparameter tuning.
  """
    trialId = _messages.IntegerField(1)