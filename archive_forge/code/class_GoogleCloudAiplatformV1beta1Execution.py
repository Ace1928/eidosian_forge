from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1Execution(_messages.Message):
    """Instance of a general execution.

  Enums:
    StateValueValuesEnum: The state of this Execution. This is a property of
      the Execution, and does not imply or capture any ongoing process. This
      property is managed by clients (such as Vertex AI Pipelines) and the
      system does not prescribe or check the validity of state transitions.

  Messages:
    LabelsValue: The labels with user-defined metadata to organize your
      Executions. Label keys and values can be no longer than 64 characters
      (Unicode codepoints), can only contain lowercase letters, numeric
      characters, underscores and dashes. International characters are
      allowed. No more than 64 user labels can be associated with one
      Execution (System labels are excluded).
    MetadataValue: Properties of the Execution. Top level metadata keys'
      heading and trailing spaces will be trimmed. The size of this field
      should not exceed 200KB.

  Fields:
    createTime: Output only. Timestamp when this Execution was created.
    description: Description of the Execution
    displayName: User provided display name of the Execution. May be up to 128
      Unicode characters.
    etag: An eTag used to perform consistent read-modify-write updates. If not
      set, a blind "overwrite" update happens.
    labels: The labels with user-defined metadata to organize your Executions.
      Label keys and values can be no longer than 64 characters (Unicode
      codepoints), can only contain lowercase letters, numeric characters,
      underscores and dashes. International characters are allowed. No more
      than 64 user labels can be associated with one Execution (System labels
      are excluded).
    metadata: Properties of the Execution. Top level metadata keys' heading
      and trailing spaces will be trimmed. The size of this field should not
      exceed 200KB.
    name: Output only. The resource name of the Execution.
    schemaTitle: The title of the schema describing the metadata. Schema title
      and version is expected to be registered in earlier Create Schema calls.
      And both are used together as unique identifiers to identify schemas
      within the local metadata store.
    schemaVersion: The version of the schema in `schema_title` to use. Schema
      title and version is expected to be registered in earlier Create Schema
      calls. And both are used together as unique identifiers to identify
      schemas within the local metadata store.
    state: The state of this Execution. This is a property of the Execution,
      and does not imply or capture any ongoing process. This property is
      managed by clients (such as Vertex AI Pipelines) and the system does not
      prescribe or check the validity of state transitions.
    updateTime: Output only. Timestamp when this Execution was last updated.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The state of this Execution. This is a property of the Execution, and
    does not imply or capture any ongoing process. This property is managed by
    clients (such as Vertex AI Pipelines) and the system does not prescribe or
    check the validity of state transitions.

    Values:
      STATE_UNSPECIFIED: Unspecified Execution state
      NEW: The Execution is new
      RUNNING: The Execution is running
      COMPLETE: The Execution has finished running
      FAILED: The Execution has failed
      CACHED: The Execution completed through Cache hit.
      CANCELLED: The Execution was cancelled.
    """
        STATE_UNSPECIFIED = 0
        NEW = 1
        RUNNING = 2
        COMPLETE = 3
        FAILED = 4
        CACHED = 5
        CANCELLED = 6

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """The labels with user-defined metadata to organize your Executions.
    Label keys and values can be no longer than 64 characters (Unicode
    codepoints), can only contain lowercase letters, numeric characters,
    underscores and dashes. International characters are allowed. No more than
    64 user labels can be associated with one Execution (System labels are
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
        """Properties of the Execution. Top level metadata keys' heading and
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