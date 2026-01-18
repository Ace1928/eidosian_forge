from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SchemaExtension(_messages.Message):
    """Represents a Managed Microsoft Identities Schema Extension.

  Enums:
    StateValueValuesEnum: Output only. The current state of the Schema
      Extension.

  Fields:
    backup: Output only. Id for backup taken before extending domain schema.
    createTime: Output only. The time the schema extension was created.
    description: Description for Schema Change.
    fileContents: File uploaded as a byte stream input.
    gcsPath: File stored in Cloud Storage bucket and represented in the form
      projects/{project_id}/buckets/{bucket_name}/objects/{object_name}
    name: The unique name of the Schema Extension in the form of projects/{pro
      ject_id}/locations/global/domains/{domain_name}/schemaExtensions/{schema
      _extension}
    state: Output only. The current state of the Schema Extension.
    statusMessage: Output only. Additional information about the current
      status of this Schema Extension, if available.
    updateTime: Output only. Last update time.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state of the Schema Extension.

    Values:
      STATE_UNSPECIFIED: Not set.
      CREATING: LDIF is currently getting applied on domain.
      COMPLETED: LDIF has been successfully applied on domain.
      FAILED: LDIF did not applied successfully.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        COMPLETED = 2
        FAILED = 3
    backup = _messages.StringField(1)
    createTime = _messages.StringField(2)
    description = _messages.StringField(3)
    fileContents = _messages.BytesField(4)
    gcsPath = _messages.StringField(5)
    name = _messages.StringField(6)
    state = _messages.EnumField('StateValueValuesEnum', 7)
    statusMessage = _messages.StringField(8)
    updateTime = _messages.StringField(9)