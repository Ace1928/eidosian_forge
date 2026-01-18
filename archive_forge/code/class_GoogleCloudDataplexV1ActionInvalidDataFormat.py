from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1ActionInvalidDataFormat(_messages.Message):
    """Action details for invalid or unsupported data files detected by
  discovery.

  Fields:
    expectedFormat: The expected data format of the entity.
    newFormat: The new unexpected data format within the entity.
    sampledDataLocations: The list of data locations sampled and used for
      format/schema inference.
  """
    expectedFormat = _messages.StringField(1)
    newFormat = _messages.StringField(2)
    sampledDataLocations = _messages.StringField(3, repeated=True)