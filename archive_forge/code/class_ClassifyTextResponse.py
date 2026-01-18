from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClassifyTextResponse(_messages.Message):
    """The document classification response message.

  Fields:
    categories: Categories representing the input document.
  """
    categories = _messages.MessageField('ClassificationCategory', 1, repeated=True)