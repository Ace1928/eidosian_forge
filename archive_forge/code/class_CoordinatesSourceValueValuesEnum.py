from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CoordinatesSourceValueValuesEnum(_messages.Enum):
    """Source from which Building.coordinates are derived.

    Values:
      CLIENT_SPECIFIED: Building.coordinates are set to the coordinates
        included in the request.
      RESOLVED_FROM_ADDRESS: Building.coordinates are automatically populated
        based on the postal address.
      SOURCE_UNSPECIFIED: Defaults to RESOLVED_FROM_ADDRESS if postal address
        is provided. Otherwise, defaults to CLIENT_SPECIFIED if coordinates
        are provided.
    """
    CLIENT_SPECIFIED = 0
    RESOLVED_FROM_ADDRESS = 1
    SOURCE_UNSPECIFIED = 2