from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatasetInputConfig(_messages.Message):
    """Input configuration for datasets.

  Fields:
    inputFiles: Files containing the sentence pairs to be imported to the
      dataset.
  """
    inputFiles = _messages.MessageField('InputFile', 1, repeated=True)