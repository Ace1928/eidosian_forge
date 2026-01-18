from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Geolocation(_messages.Message):
    """Represents a geographical location for a given access.

  Fields:
    regionCode: A CLDR.
  """
    regionCode = _messages.StringField(1)