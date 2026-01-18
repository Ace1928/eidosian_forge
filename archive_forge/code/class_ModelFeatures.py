from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ModelFeatures(_messages.Message):
    """Represents the collection of features belonging to a model

  Fields:
    modelFeature: Repeated field that contains all features of the model
  """
    modelFeature = _messages.MessageField('ModelFeature', 1, repeated=True)