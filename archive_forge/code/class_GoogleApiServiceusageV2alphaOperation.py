from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiServiceusageV2alphaOperation(_messages.Message):
    """A GoogleApiServiceusageV2alphaOperation object.

  Fields:
    name: Name of the operation.
    requestMessageType: The type to use when sending requests. For example,
      `CreateBookRequest`.
    responseMessageType: The type that will be sent with responses. For
      example, `Book`.
  """
    name = _messages.StringField(1)
    requestMessageType = _messages.StringField(2)
    responseMessageType = _messages.StringField(3)