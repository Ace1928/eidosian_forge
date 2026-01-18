from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2Participant(_messages.Message):
    """Represents a conversation participant (human agent, virtual agent, end-
  user).

  Enums:
    RoleValueValuesEnum: Immutable. The role this participant plays in the
      conversation. This field must be set during participant creation and is
      then immutable.

  Messages:
    DocumentsMetadataFiltersValue: Optional. Key-value filters on the metadata
      of documents returned by article suggestion. If specified, article
      suggestion only returns suggested documents that match all filters in
      their Document.metadata. Multiple values for a metadata key should be
      concatenated by comma. For example, filters to match all documents that
      have 'US' or 'CA' in their market metadata values and 'agent' in their
      user metadata values will be ``` documents_metadata_filters { key:
      "market" value: "US,CA" } documents_metadata_filters { key: "user"
      value: "agent" } ```

  Fields:
    documentsMetadataFilters: Optional. Key-value filters on the metadata of
      documents returned by article suggestion. If specified, article
      suggestion only returns suggested documents that match all filters in
      their Document.metadata. Multiple values for a metadata key should be
      concatenated by comma. For example, filters to match all documents that
      have 'US' or 'CA' in their market metadata values and 'agent' in their
      user metadata values will be ``` documents_metadata_filters { key:
      "market" value: "US,CA" } documents_metadata_filters { key: "user"
      value: "agent" } ```
    name: Optional. The unique identifier of this participant. Format:
      `projects//locations//conversations//participants/`.
    obfuscatedExternalUserId: Optional. Obfuscated user id that should be
      associated with the created participant. You can specify a user id as
      follows: 1. If you set this field in CreateParticipantRequest or
      UpdateParticipantRequest, Dialogflow adds the obfuscated user id with
      the participant. 2. If you set this field in AnalyzeContent or
      StreamingAnalyzeContent, Dialogflow will update
      Participant.obfuscated_external_user_id. Dialogflow returns an error if
      you try to add a user id for a non-END_USER participant. Dialogflow uses
      this user id for billing and measurement purposes. For example,
      Dialogflow determines whether a user in one conversation returned in a
      later conversation. Note: * Please never pass raw user ids to
      Dialogflow. Always obfuscate your user id first. * Dialogflow only
      accepts a UTF-8 encoded string, e.g., a hex digest of a hash function
      like SHA-512. * The length of the user id must be <= 256 characters.
    role: Immutable. The role this participant plays in the conversation. This
      field must be set during participant creation and is then immutable.
    sipRecordingMediaLabel: Optional. Label applied to streams representing
      this participant in SIPREC XML metadata and SDP. This is used to assign
      transcriptions from that media stream to this participant. This field
      can be updated.
  """

    class RoleValueValuesEnum(_messages.Enum):
        """Immutable. The role this participant plays in the conversation. This
    field must be set during participant creation and is then immutable.

    Values:
      ROLE_UNSPECIFIED: Participant role not set.
      HUMAN_AGENT: Participant is a human agent.
      AUTOMATED_AGENT: Participant is an automated agent, such as a Dialogflow
        agent.
      END_USER: Participant is an end user that has called or chatted with
        Dialogflow services.
    """
        ROLE_UNSPECIFIED = 0
        HUMAN_AGENT = 1
        AUTOMATED_AGENT = 2
        END_USER = 3

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DocumentsMetadataFiltersValue(_messages.Message):
        """Optional. Key-value filters on the metadata of documents returned by
    article suggestion. If specified, article suggestion only returns
    suggested documents that match all filters in their Document.metadata.
    Multiple values for a metadata key should be concatenated by comma. For
    example, filters to match all documents that have 'US' or 'CA' in their
    market metadata values and 'agent' in their user metadata values will be
    ``` documents_metadata_filters { key: "market" value: "US,CA" }
    documents_metadata_filters { key: "user" value: "agent" } ```

    Messages:
      AdditionalProperty: An additional property for a
        DocumentsMetadataFiltersValue object.

    Fields:
      additionalProperties: Additional properties of type
        DocumentsMetadataFiltersValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DocumentsMetadataFiltersValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    documentsMetadataFilters = _messages.MessageField('DocumentsMetadataFiltersValue', 1)
    name = _messages.StringField(2)
    obfuscatedExternalUserId = _messages.StringField(3)
    role = _messages.EnumField('RoleValueValuesEnum', 4)
    sipRecordingMediaLabel = _messages.StringField(5)