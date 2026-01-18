from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Projection(_messages.Message):
    """The projection of document's fields to return.

  Fields:
    fields: The fields to return. If empty, all fields are returned. To only
      return the name of the document, use `['__name__']`.
  """
    fields = _messages.MessageField('FieldReference', 1, repeated=True)