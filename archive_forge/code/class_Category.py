from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Category(_messages.Message):
    """The category to which the update belongs.

  Fields:
    categoryId: The identifier of the category.
    name: The localized name of the category.
  """
    categoryId = _messages.StringField(1)
    name = _messages.StringField(2)