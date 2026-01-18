from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DocumentTransform(_messages.Message):
    """A transformation of a document.

  Fields:
    document: The name of the document to transform.
    fieldTransforms: The list of transformations to apply to the fields of the
      document, in order. This must not be empty.
  """
    document = _messages.StringField(1)
    fieldTransforms = _messages.MessageField('FieldTransform', 2, repeated=True)