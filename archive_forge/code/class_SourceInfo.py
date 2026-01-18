from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourceInfo(_messages.Message):
    """Source information used to create a Service Config

  Messages:
    SourceFilesValueListEntry: A SourceFilesValueListEntry object.

  Fields:
    sourceFiles: All files used during config generation.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class SourceFilesValueListEntry(_messages.Message):
        """A SourceFilesValueListEntry object.

    Messages:
      AdditionalProperty: An additional property for a
        SourceFilesValueListEntry object.

    Fields:
      additionalProperties: Properties of the object. Contains field @type
        with type URL.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a SourceFilesValueListEntry object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    sourceFiles = _messages.MessageField('SourceFilesValueListEntry', 1, repeated=True)