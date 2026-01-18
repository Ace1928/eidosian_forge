from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudAiLargeModelsVisionRaiInfo(_messages.Message):
    """A CloudAiLargeModelsVisionRaiInfo object.

  Fields:
    raiCategories: List of rai categories' information to return
    scores: List of rai scores mapping to the rai categories. Rounded to 1
      decimal place.
  """
    raiCategories = _messages.StringField(1, repeated=True)
    scores = _messages.FloatField(2, repeated=True, variant=_messages.Variant.FLOAT)