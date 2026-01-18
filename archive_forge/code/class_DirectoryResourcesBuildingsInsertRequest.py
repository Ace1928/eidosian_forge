from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryResourcesBuildingsInsertRequest(_messages.Message):
    """A DirectoryResourcesBuildingsInsertRequest object.

  Enums:
    CoordinatesSourceValueValuesEnum: Source from which Building.coordinates
      are derived.

  Fields:
    building: A Building resource to be passed as the request body.
    coordinatesSource: Source from which Building.coordinates are derived.
    customer: The unique ID for the customer's G Suite account. As an account
      administrator, you can also use the my_customer alias to represent your
      account's customer ID.
  """

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
    building = _messages.MessageField('Building', 1)
    coordinatesSource = _messages.EnumField('CoordinatesSourceValueValuesEnum', 2, default=u'SOURCE_UNSPECIFIED')
    customer = _messages.StringField(3, required=True)