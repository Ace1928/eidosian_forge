from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CategoriesValue(_messages.Message):
    """Deprecated.

    Fields:
      names: Deprecated.
    """
    names = _messages.StringField(1, repeated=True)