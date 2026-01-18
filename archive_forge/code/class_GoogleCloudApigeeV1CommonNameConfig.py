from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1CommonNameConfig(_messages.Message):
    """A GoogleCloudApigeeV1CommonNameConfig object.

  Fields:
    matchWildCards: A boolean attribute.
    name: A string attribute.
  """
    matchWildCards = _messages.BooleanField(1)
    name = _messages.StringField(2)