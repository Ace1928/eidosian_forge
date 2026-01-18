from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1TimestampSplit(_messages.Message):
    """Assigns input data to training, validation, and test sets based on a
  provided timestamps. The youngest data pieces are assigned to training set,
  next to validation set, and the oldest to the test set. Supported only for
  tabular Datasets.

  Fields:
    key: Required. The key is a name of one of the Dataset's data columns. The
      values of the key (the values in the column) must be in RFC 3339 `date-
      time` format, where `time-offset` = `"Z"` (e.g.
      1985-04-12T23:20:50.52Z). If for a piece of data the key is not present
      or has an invalid value, that piece is ignored by the pipeline.
    testFraction: The fraction of the input data that is to be used to
      evaluate the Model.
    trainingFraction: The fraction of the input data that is to be used to
      train the Model.
    validationFraction: The fraction of the input data that is to be used to
      validate the Model.
  """
    key = _messages.StringField(1)
    testFraction = _messages.FloatField(2)
    trainingFraction = _messages.FloatField(3)
    validationFraction = _messages.FloatField(4)