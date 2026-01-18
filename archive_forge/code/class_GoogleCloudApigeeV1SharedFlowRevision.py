from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1SharedFlowRevision(_messages.Message):
    """The metadata describing a shared flow revision.

  Messages:
    EntityMetaDataAsPropertiesValue: A Key-Value map of metadata about this
      shared flow revision.

  Fields:
    configurationVersion: The version of the configuration schema to which
      this shared flow conforms. The only supported value currently is
      majorVersion 4 and minorVersion 0. This setting may be used in the
      future to enable evolution of the shared flow format.
    contextInfo: A textual description of the shared flow revision.
    createdAt: Time at which this shared flow revision was created, in
      milliseconds since epoch.
    description: Description of the shared flow revision.
    displayName: The human readable name of this shared flow.
    entityMetaDataAsProperties: A Key-Value map of metadata about this shared
      flow revision.
    lastModifiedAt: Time at which this shared flow revision was most recently
      modified, in milliseconds since epoch.
    name: The resource ID of the parent shared flow.
    policies: A list of policy names included in this shared flow revision.
    resourceFiles: The resource files included in this shared flow revision.
    resources: A list of the resources included in this shared flow revision
      formatted as "{type}://{name}".
    revision: The resource ID of this revision.
    sharedFlows: A list of the shared flow names included in this shared flow
      revision.
    type: The string "Application"
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class EntityMetaDataAsPropertiesValue(_messages.Message):
        """A Key-Value map of metadata about this shared flow revision.

    Messages:
      AdditionalProperty: An additional property for a
        EntityMetaDataAsPropertiesValue object.

    Fields:
      additionalProperties: Additional properties of type
        EntityMetaDataAsPropertiesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a EntityMetaDataAsPropertiesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    configurationVersion = _messages.MessageField('GoogleCloudApigeeV1ConfigVersion', 1)
    contextInfo = _messages.StringField(2)
    createdAt = _messages.IntegerField(3)
    description = _messages.StringField(4)
    displayName = _messages.StringField(5)
    entityMetaDataAsProperties = _messages.MessageField('EntityMetaDataAsPropertiesValue', 6)
    lastModifiedAt = _messages.IntegerField(7)
    name = _messages.StringField(8)
    policies = _messages.StringField(9, repeated=True)
    resourceFiles = _messages.MessageField('GoogleCloudApigeeV1ResourceFiles', 10)
    resources = _messages.StringField(11, repeated=True)
    revision = _messages.StringField(12)
    sharedFlows = _messages.StringField(13, repeated=True)
    type = _messages.StringField(14)