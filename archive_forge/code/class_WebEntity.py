from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WebEntity(_messages.Message):
    """Entity deduced from similar images on the Internet.

  Fields:
    description: Canonical description of the entity, in English.
    entityId: Opaque entity ID.
    score: Overall relevancy score for the entity. Not normalized and not
      comparable across different image queries.
  """
    description = _messages.StringField(1)
    entityId = _messages.StringField(2)
    score = _messages.FloatField(3, variant=_messages.Variant.FLOAT)