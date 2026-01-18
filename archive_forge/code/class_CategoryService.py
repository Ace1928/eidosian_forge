from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CategoryService(_messages.Message):
    """A service that belongs to a category and its metadata.

  Fields:
    parent: The parent category to which this service belongs.
    service: Output only. The service included by this category.
  """
    parent = _messages.StringField(1)
    service = _messages.MessageField('Service', 2)