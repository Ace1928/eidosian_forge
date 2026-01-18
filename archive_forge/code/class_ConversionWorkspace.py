from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConversionWorkspace(_messages.Message):
    """The main conversion workspace resource entity.

  Messages:
    GlobalSettingsValue: Optional. A generic list of settings for the
      workspace. The settings are database pair dependant and can indicate
      default behavior for the mapping rules engine or turn on or off specific
      features. Such examples can be: convert_foreign_key_to_interleave=true,
      skip_triggers=false, ignore_non_table_synonyms=true

  Fields:
    createTime: Output only. The timestamp when the workspace resource was
      created.
    destination: Required. The destination engine details.
    displayName: Optional. The display name for the workspace.
    globalSettings: Optional. A generic list of settings for the workspace.
      The settings are database pair dependant and can indicate default
      behavior for the mapping rules engine or turn on or off specific
      features. Such examples can be: convert_foreign_key_to_interleave=true,
      skip_triggers=false, ignore_non_table_synonyms=true
    hasUncommittedChanges: Output only. Whether the workspace has uncommitted
      changes (changes which were made after the workspace was committed).
    latestCommitId: Output only. The latest commit ID.
    latestCommitTime: Output only. The timestamp when the workspace was
      committed.
    name: Full name of the workspace resource, in the form of: projects/{proje
      ct}/locations/{location}/conversionWorkspaces/{conversion_workspace}.
    source: Required. The source engine details.
    updateTime: Output only. The timestamp when the workspace resource was
      last updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class GlobalSettingsValue(_messages.Message):
        """Optional. A generic list of settings for the workspace. The settings
    are database pair dependant and can indicate default behavior for the
    mapping rules engine or turn on or off specific features. Such examples
    can be: convert_foreign_key_to_interleave=true, skip_triggers=false,
    ignore_non_table_synonyms=true

    Messages:
      AdditionalProperty: An additional property for a GlobalSettingsValue
        object.

    Fields:
      additionalProperties: Additional properties of type GlobalSettingsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a GlobalSettingsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    createTime = _messages.StringField(1)
    destination = _messages.MessageField('DatabaseEngineInfo', 2)
    displayName = _messages.StringField(3)
    globalSettings = _messages.MessageField('GlobalSettingsValue', 4)
    hasUncommittedChanges = _messages.BooleanField(5)
    latestCommitId = _messages.StringField(6)
    latestCommitTime = _messages.StringField(7)
    name = _messages.StringField(8)
    source = _messages.MessageField('DatabaseEngineInfo', 9)
    updateTime = _messages.StringField(10)