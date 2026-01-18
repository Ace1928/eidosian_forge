from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class RequestValue(_messages.Message):
    """The schema for the request.

    Fields:
      _ref: Schema ID for the request schema.
    """
    _ref = _messages.StringField(1)