from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WebImage(_messages.Message):
    """Metadata for online images.

  Fields:
    score: (Deprecated) Overall relevancy score for the image.
    url: The result image URL.
  """
    score = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    url = _messages.StringField(2)