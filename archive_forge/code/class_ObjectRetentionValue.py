from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ObjectRetentionValue(_messages.Message):
    """The bucket's object retention config.

    Fields:
      mode: The bucket's object retention mode. Can be Enabled.
    """
    mode = _messages.StringField(1)