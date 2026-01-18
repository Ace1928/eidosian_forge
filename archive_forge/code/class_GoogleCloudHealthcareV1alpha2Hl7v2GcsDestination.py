from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudHealthcareV1alpha2Hl7v2GcsDestination(_messages.Message):
    """The Cloud Storage output destination. The Cloud Healthcare Service Agent
  requires the `roles/storage.objectAdmin` Cloud IAM roles on the Cloud
  Storage location.

  Enums:
    ContentStructureValueValuesEnum: The format of the exported HL7v2 message
      files.
    MessageViewValueValuesEnum: Specifies the parts of the Message resource to
      include in the export. If not specified, FULL is used.

  Fields:
    contentStructure: The format of the exported HL7v2 message files.
    messageView: Specifies the parts of the Message resource to include in the
      export. If not specified, FULL is used.
    uriPrefix: URI for a Cloud Storage directory where the server writes
      result files, in the format `gs://{bucket-
      id}/{path/to/destination/dir}`. If there is no trailing slash, the
      service appends one when composing the object path. The user is
      responsible for creating the Cloud Storage bucket referenced in
      `uri_prefix`.
  """

    class ContentStructureValueValuesEnum(_messages.Enum):
        """The format of the exported HL7v2 message files.

    Values:
      CONTENT_STRUCTURE_UNSPECIFIED: If the content structure is not
        specified, the default value `MESSAGE_JSON` will be used.
      MESSAGE_JSON: Messages are printed using the JSON format returned from
        the `GetMessage` API. Messages are delimited with newlines.
    """
        CONTENT_STRUCTURE_UNSPECIFIED = 0
        MESSAGE_JSON = 1

    class MessageViewValueValuesEnum(_messages.Enum):
        """Specifies the parts of the Message resource to include in the export.
    If not specified, FULL is used.

    Values:
      MESSAGE_VIEW_UNSPECIFIED: Not specified, equivalent to FULL for
        getMessage, equivalent to BASIC for listMessages.
      RAW_ONLY: Server responses include all the message fields except
        parsed_data and schematized_data fields.
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
    contentStructure = _messages.EnumField('ContentStructureValueValuesEnum', 1)
    messageView = _messages.EnumField('MessageViewValueValuesEnum', 2)
    uriPrefix = _messages.StringField(3)