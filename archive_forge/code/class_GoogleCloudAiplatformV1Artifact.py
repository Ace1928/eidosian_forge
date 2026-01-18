from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1Artifact(_messages.Message):
    """Instance of a general artifact.

  Enums:
    StateValueValuesEnum: The state of this Artifact. This is a property of
      the Artifact, and does not imply or capture any ongoing process. This
      property is managed by clients (such as Vertex AI Pipelines), and the
      system does not prescribe or check the validity of state transitions.

  Messages:
    LabelsValue: The labels with user-defined metadata to organize your
      Artifacts. Label keys and values can be no longer than 64 characters
      (Unicode codepoints), can only contain lowercase letters, numeric
      characters, underscores and dashes. International characters are
      allowed. No more than 64 user labels can be associated with one Artifact
      (System labels are excluded).
    MetadataValue: Properties of the Artifact. Top level metadata keys'
      heading and trailing spaces will be trimmed. The size of this field
      should not exceed 200KB.

  Fields:
    createTime: Output only. Timestamp when this Artifact was created.
    description: Description of the Artifact
    displayName: User provided display name of the Artifact. May be up to 128
      Unicode characters.
    etag: An eTag used to perform consistent read-modify-write updates. If not
      set, a blind "overwrite" update happens.
    labels: The labels with user-defined metadata to organize your Artifacts.
      Label keys and values can be no longer than 64 characters (Unicode
      codepoints), can only contain lowercase letters, numeric characters,
      underscores and dashes. International characters are allowed. No more
      than 64 user labels can be associated with one Artifact (System labels
      are excluded).
    metadata: Properties of the Artifact. Top level metadata keys' heading and
      trailing spaces will be trimmed. The size of this field should not
      exceed 200KB.
    name: Output only. The resource name of the Artifact.
    schemaTitle: The title of the schema describing the metadata. Schema title
      and version is expected to be registered in earlier Create Schema calls.
      And both are used together as unique identifiers to identify schemas
      within the local metadata store.
    schemaVersion: The version of the schema in schema_name to use. Schema
      title and version is expected to be registered in earlier Create Schema
      calls. And both are used together as unique identifiers to identify
      schemas within the local metadata store.
    state: The state of this Artifact. This is a property of the Artifact, and
      does not imply or capture any ongoing process. This property is managed
      by clients (such as Vertex AI Pipelines), and the system does not
      prescribe or check the validity of state transitions.
    updateTime: Output only. Timestamp when this Artifact was last updated.
    uri: The uniform resource identifier of the artifact file. May be empty if
      there is no actual artifact file.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The state of this Artifact. This is a property of the Artifact, and
    does not imply or capture any ongoing process. This property is managed by
    clients (such as Vertex AI Pipelines), and the system does not prescribe
    or check the validity of state transitions.

    Values:
      STATE_UNSPECIFIED: Unspecified state for the Artifact.
      PENDING: A state used by systems like Vertex AI Pipelines to indicate
        that the underlying data item represented by this Artifact is being
        created.
      LIVE: A state indicating that the Artifact should exist, unless
        something external to the system deletes it.
    """
        STATE_UNSPECIFIED = 0
        PENDING = 1
        LIVE = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """The labels with user-defined metadata to organize your Artifacts.
    Label keys and values can be no longer than 64 characters (Unicode
    codepoints), can only contain lowercase letters, numeric characters,
    underscores and dashes. International characters are allowed. No more than
    64 user labels can be associated with one Artifact (System labels are
    excluded).

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MetadataValue(_messages.Message):
        """Properties of the Artifact. Top level metadata keys' heading and
    trailing spaces will be trimmed. The size of this field should not exceed
    200KB.

    Messages:
      AdditionalProperty: An additional property for a MetadataValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MetadataValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)
    etag = _messages.StringField(4)
    labels = _messages.MessageField('LabelsValue', 5)
    metadata = _messages.MessageField('MetadataValue', 6)
    name = _messages.StringField(7)
    schemaTitle = _messages.StringField(8)
    schemaVersion = _messages.StringField(9)
    state = _messages.EnumField('StateValueValuesEnum', 10)
    updateTime = _messages.StringField(11)
    uri = _messages.StringField(12)