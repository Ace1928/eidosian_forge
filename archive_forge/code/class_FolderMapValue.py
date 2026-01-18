from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class FolderMapValue(_messages.Message):
    """A map of folder id and folder config to specify consumer projects for
    this shared-reservation. This is only valid when share_type's value is
    DIRECT_PROJECTS_UNDER_SPECIFIC_FOLDERS. Folder id should be a string of
    number, and without "folders/" prefix.

    Messages:
      AdditionalProperty: An additional property for a FolderMapValue object.

    Fields:
      additionalProperties: Additional properties of type FolderMapValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a FolderMapValue object.

      Fields:
        key: Name of the additional property.
        value: A ShareSettingsFolderConfig attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('ShareSettingsFolderConfig', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)