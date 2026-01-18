from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsHl7V2StoresMessagesGetRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsHl7V2StoresMessagesGetRequest
  object.

  Enums:
    ViewValueValuesEnum: Specifies which parts of the Message resource to
      return in the response. When unspecified, equivalent to FULL.

  Fields:
    name: The resource name of the HL7v2 message to retrieve.
    view: Specifies which parts of the Message resource to return in the
      response. When unspecified, equivalent to FULL.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Specifies which parts of the Message resource to return in the
    response. When unspecified, equivalent to FULL.

    Values:
      MESSAGE_VIEW_UNSPECIFIED: Not specified, equivalent to FULL.
      RAW_ONLY: Server responses include all the message fields except
        parsed_data field, and schematized_data fields.
      PARSED_ONLY: Server responses include all the message fields except data
        field, and schematized_data fields.
      FULL: Server responses include all the message fields.
      SCHEMATIZED_ONLY: Server responses include all the message fields except
        data and parsed_data fields.
      BASIC: Server responses include only the name field.
    """
        MESSAGE_VIEW_UNSPECIFIED = 0
        RAW_ONLY = 1
        PARSED_ONLY = 2
        FULL = 3
        SCHEMATIZED_ONLY = 4
        BASIC = 5
    name = _messages.StringField(1, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 2)