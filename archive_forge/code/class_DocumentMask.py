from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DocumentMask(_messages.Message):
    """A set of field paths on a document. Used to restrict a get or update
  operation on a document to a subset of its fields. This is different from
  standard field masks, as this is always scoped to a Document, and takes in
  account the dynamic nature of Value.

  Fields:
    fieldPaths: The list of field paths in the mask. See Document.fields for a
      field path syntax reference.
  """
    fieldPaths = _messages.StringField(1, repeated=True)