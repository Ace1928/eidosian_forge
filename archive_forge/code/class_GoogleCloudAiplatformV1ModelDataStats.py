from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ModelDataStats(_messages.Message):
    """Stats of data used for train or evaluate the Model.

  Fields:
    testAnnotationsCount: Number of Annotations that are used for evaluating
      this Model. If the Model is evaluated multiple times, this will be the
      number of test Annotations used by the first evaluation. If the Model is
      not evaluated, the number is 0.
    testDataItemsCount: Number of DataItems that were used for evaluating this
      Model. If the Model is evaluated multiple times, this will be the number
      of test DataItems used by the first evaluation. If the Model is not
      evaluated, the number is 0.
    trainingAnnotationsCount: Number of Annotations that are used for training
      this Model.
    trainingDataItemsCount: Number of DataItems that were used for training
      this Model.
    validationAnnotationsCount: Number of Annotations that are used for
      validating this Model during training.
    validationDataItemsCount: Number of DataItems that were used for
      validating this Model during training.
  """
    testAnnotationsCount = _messages.IntegerField(1)
    testDataItemsCount = _messages.IntegerField(2)
    trainingAnnotationsCount = _messages.IntegerField(3)
    trainingDataItemsCount = _messages.IntegerField(4)
    validationAnnotationsCount = _messages.IntegerField(5)
    validationDataItemsCount = _messages.IntegerField(6)