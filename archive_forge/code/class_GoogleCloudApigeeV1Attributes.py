from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1Attributes(_messages.Message):
    """A GoogleCloudApigeeV1Attributes object.

  Fields:
    attribute: List of attributes.
  """
    attribute = _messages.MessageField('GoogleCloudApigeeV1Attribute', 1, repeated=True)