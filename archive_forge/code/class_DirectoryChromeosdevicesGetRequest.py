from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryChromeosdevicesGetRequest(_messages.Message):
    """A DirectoryChromeosdevicesGetRequest object.

  Enums:
    ProjectionValueValuesEnum: Restrict information returned to a set of
      selected fields.

  Fields:
    customerId: Immutable ID of the G Suite account
    deviceId: Immutable ID of Chrome OS Device
    projection: Restrict information returned to a set of selected fields.
  """

    class ProjectionValueValuesEnum(_messages.Enum):
        """Restrict information returned to a set of selected fields.

    Values:
      BASIC: Includes only the basic metadata fields (e.g., deviceId,
        serialNumber, status, and user)
      FULL: Includes all metadata fields
    """
        BASIC = 0
        FULL = 1
    customerId = _messages.StringField(1, required=True)
    deviceId = _messages.StringField(2, required=True)
    projection = _messages.EnumField('ProjectionValueValuesEnum', 3)