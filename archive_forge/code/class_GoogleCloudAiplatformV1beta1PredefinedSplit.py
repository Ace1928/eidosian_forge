from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1PredefinedSplit(_messages.Message):
    """Assigns input data to training, validation, and test sets based on the
  value of a provided key. Supported only for tabular Datasets.

  Fields:
    key: Required. The key is a name of one of the Dataset's data columns. The
      value of the key (either the label's value or value in the column) must
      be one of {`training`, `validation`, `test`}, and it defines to which
      set the given piece of data is assigned. If for a piece of data the key
      is not present or has an invalid value, that piece is ignored by the
      pipeline.
  """
    key = _messages.StringField(1)