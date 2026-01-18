from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ExportFractionSplit(_messages.Message):
    """Assigns the input data to training, validation, and test sets as per the
  given fractions. Any of `training_fraction`, `validation_fraction` and
  `test_fraction` may optionally be provided, they must sum to up to 1. If the
  provided ones sum to less than 1, the remainder is assigned to sets as
  decided by Vertex AI. If none of the fractions are set, by default roughly
  80% of data is used for training, 10% for validation, and 10% for test.

  Fields:
    testFraction: The fraction of the input data that is to be used to
      evaluate the Model.
    trainingFraction: The fraction of the input data that is to be used to
      train the Model.
    validationFraction: The fraction of the input data that is to be used to
      validate the Model.
  """
    testFraction = _messages.FloatField(1)
    trainingFraction = _messages.FloatField(2)
    validationFraction = _messages.FloatField(3)