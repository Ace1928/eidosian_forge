from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsHl7V2StoresMessagesBatchGetRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsHl7V2StoresMessagesBatchGetRequest
  object.

  Enums:
    ViewValueValuesEnum: Specifies the parts of the Messages resource to
      return in the response. When unspecified, equivalent to BASIC.

  Fields:
    ids: The resource id of the HL7v2 messages to retrieve in the format:
      `{message_id}`, where the full resource name is
      `{parent}/messages/{message_id}` A maximum of 100 messages can be
      retrieved in a batch. All 'ids' have to be under parent.
    parent: Required. Name of the HL7v2 store to retrieve messages from, in
      the format: `projects/{project_id}/locations/{location_id}/datasets/{dat
      aset_id}/hl7v2Stores/{hl7v2_store_id}`.
    view: Specifies the parts of the Messages resource to return in the
      response. When unspecified, equivalent to BASIC.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Specifies the parts of the Messages resource to return in the
    response. When unspecified, equivalent to BASIC.

    Values:
      MESSAGE_VIEW_UNSPECIFIED: Not specified, equivalent to FULL for
        getMessage, equivalent to BASIC for listMessages.
      RAW_ONLY: Server responses include all the message fields except
        parsed_data, and schematized_data fields.
      PARSED_ONLY: Server responses include all the message fields except data
        and schematized_data fields.
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
    ids = _messages.StringField(1, repeated=True)
    parent = _messages.StringField(2, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 3)