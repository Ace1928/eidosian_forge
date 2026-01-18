from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BuildingCoordinates(_messages.Message):
    """JSON template for coordinates of a building in Directory API.

  Fields:
    latitude: Latitude in decimal degrees.
    longitude: Longitude in decimal degrees.
  """
    latitude = _messages.FloatField(1)
    longitude = _messages.FloatField(2)