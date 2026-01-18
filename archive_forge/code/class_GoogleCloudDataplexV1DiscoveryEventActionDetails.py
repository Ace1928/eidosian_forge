from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DiscoveryEventActionDetails(_messages.Message):
    """Details about the action.

  Fields:
    type: The type of action. Eg. IncompatibleDataSchema, InvalidDataFormat
  """
    type = _messages.StringField(1)